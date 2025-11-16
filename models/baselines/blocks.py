import math
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat
# from mamba_ssm import Mamba
import torch.utils.checkpoint as checkpoint
import pytorch_wavelets as wavelets
import cv2



# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=1):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out) * x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 使用深度可分离卷积（Depthwise Separable Convolution）
        self.depthwise_conv1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, groups=in_planes, bias=False)
        self.depthwise_conv2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, groups=in_planes, bias=False)
        
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.depthwise_conv2(self.relu1(self.depthwise_conv1(self.avg_pool(x))))
        max_out = self.depthwise_conv2(self.relu1(self.depthwise_conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=1, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)

    def forward(self, x):
        x = self.channelattention(x)
        return x


class MulAttBlock(nn.Module):
    def __init__(self, dim):
        super(MulAttBlock, self).__init__()
        self.dim = dim
        self.conv_resize1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=2, padding=1, bias=True, groups=4),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=2, padding=1, bias=True, groups=4),
        )
        self.conv_resize2 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1, bias=False, groups=4),
            nn.PixelShuffle(2),
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1, bias=False, groups=4),
            nn.PixelShuffle(2)
        )
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.scale = dim ** -0.5
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x_in):
        x = self.conv_resize1(x_in)

        B, C, H, W = x.shape
        x = self.conv1(x)
        x = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, 8, 8)
        x1 = self.conv2(x).reshape(-1, C, 8 * 8)
        x2 = self.conv3(x).reshape(-1, C, 8 * 8).transpose(1, 2)
        att = (x2 @ x1) * self.scale
        att = att.softmax(dim=1)
        x = (x.reshape(-1, C, 8 * 8) @ att).reshape(-1, C, 8, 8)
        x = self.conv4(x)
        x = x.reshape(B, H // 8, W // 8, C, 8, 8).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        x = self.conv_resize2(x)
        out = x_in + x
        return out

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.res = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.res(x)

class DWT_block(nn.Module):
    def __init__(self, inchannels, m1=2, m2=4,  layers=None, init=None): #dmcs=None, init=None,
        super(DWT_block, self).__init__()
        self.init = init
        self.inchannels = inchannels
        self.dim = 4
        self.m1 = m1
        self.m2 = m2
        self.x_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=16, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
        )
        self.x_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
        )
        self.x_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.ReLU(),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(1, self.dim * 8, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.dim * 8, self.dim * 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )


        self.high_conv = nn.Sequential(
            nn.Conv2d(3, self.dim, kernel_size=1, padding=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )
        self.low_conv = nn.Sequential(
            nn.Conv2d(1, self.dim, kernel_size=1, padding=1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )

        self.high_conv_ = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.low_conv_ = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, padding=1, bias=True),
            nn.ReLU()
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=6, padding=0, bias=True),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv2d(1, 1, kernel_size=1, padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=True)

        self.ca1 = cbam_block(1)
        self.ca2 = cbam_block(1)
        self.res = ResBlock(1)
        self.att = MulAttBlock(self.dim) if (layers + 1) % 4 == 0 else None
        self.dwt = wavelets.DWTForward(J=1, wave='haar', mode='zero') #初始化离散小波变换（DWT）层  

    def forward(self, x, y=None):

        f1 = self.x_conv1(x)
        f2 = self.x_conv2(f1)
        f3 = self.x_conv3(f2)

        tmp12 = f3
        coeffs = self.dwt(tmp12)
        cA, (cH, cV, cD) = coeffs[0], (coeffs[1][0][:, :, _, :, :] for _ in range(3))
        hf = self.high_conv(torch.cat((cH, cV, cD), dim=1))  
        lf = self.low_conv(cA)

        hf_np = hf.squeeze().cpu().detach().numpy() #高斯模糊 操作
        hf_blurred = cv2.GaussianBlur(hf_np, (5, 5), 0.5)
        hf = torch.from_numpy(hf_blurred).unsqueeze(0).unsqueeze(0).cuda()
        
        hf = self.ca1(hf)
        lf = self.ca2(lf)

        alpha = 1.0
        cat_hf_lf = self.cat_conv(torch.cat((alpha*hf, (1-alpha)*lf), dim=1))
        tmp12 = tmp12 + cat_hf_lf

        f12 = tmp12
        f12 = self.att(f12) if self.att is not None else f12
        f13 = self.conv9(f12)  


        return f13




# class MambaLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.norm = nn.LayerNorm(input_dim)
#         self.mamba = Mamba(
#                 d_model=input_dim, 
#                 d_state=d_state,  
#                 d_conv=d_conv,    
#                 expand=expand,    
#         )
#         self.proj = nn.Linear(input_dim, output_dim)
#         self.skip_scale= nn.Parameter(torch.ones(1))
    
#     def forward(self, x):
#         if x.dtype == torch.float16:
#             x = x.type(torch.float32)
#         B, C = x.shape[:2]  
#         assert C == self.input_dim
#         n_tokens = x.shape[2:].numel() 
#         img_dims = x.shape[2:]   
#         x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2) 
#         x_norm = self.norm(x_flat)  
#         x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat  
#         x_mamba = self.norm(x_mamba)  
#         x_mamba = self.proj(x_mamba)   
#         out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)  
#         return out

class DualPathFFN(nn.Module):
    """
    输入 x:[B,C,H,W] -> 
      Branch A: 1×1 → GELU → DWConv3×3(dil=1) → ReLU → 1×1 → [B,C,H,W]
      Branch B: 1×1 → GELU → DWConv3×3(dil=2) → ReLU → 1×1 → [B,C,H,W]
    concat([A,B], dim=1) => [B,2C,H,W] → 1×1 融合回 C → Dropout
    """
    def __init__(self, dim, expand_ratio=1.0, drop=0.0):
        super().__init__()
        hidden = int(dim * expand_ratio)

        # --- Branch A（常规感受野） ---
        self.a = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),  # DW 3×3, dil=1
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
        )

        # --- Branch B（更大感受野/稀疏感受野） ---
        self.b = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden, bias=False),  # DW 3×3, dil=2
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
        )

        # concat 后通道 2C，压回 C
        self.fuse = nn.Conv2d(2*dim, dim, kernel_size=1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        xa = self.a(x)   # [B,C,H,W]
        xb = self.b(x)   # [B,C,H,W]
        y = torch.cat([xa, xb], dim=1)   # [B,2C,H,W]
        y = self.fuse(y)                 # [B,C,H,W]
        y = self.drop(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm_attn = nn.GroupNorm(1, dim)  #类似layernorm
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # self.norm2 = nn.LayerNorm(dim)
        
        # Replace MLP with DualPathFFN
        self.dualpathffn = DualPathFFN(dim, expand_ratio=mlp_ratio, drop=drop)
    
    def forward(self, x):
        B, C, H, W = x.shape
        print("TransformerBlock input x.shape:", x.shape)
        x_norm1 = self.norm_attn(x)  # [B, C, H, W]
        print("TransformerBlock x_norm1.shape:", x_norm1.shape)
        # exit("PPM")
        x_flat = x.flatten(2).permute(0, 2, 1)  # B, N, C
        
        # 自注意力
        attn_out = self.attn(x_flat, x_flat, x_flat)[0]
        x_flat = x_flat + attn_out
        
        # Apply DualPathFFN (two-branch structure)
        x_flat = x_flat.permute(0, 2, 1).reshape(B, C, H, W)  # Convert back to [B, C, H, W]
        x_flat = self.dualpathffn(x_flat) + x # Apply DualPathFFN to replace MLP
        
        return x_flat



class SF_Block(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path, H, W, mlp_ratio=2, dim=64, m1=2, m2=4#256,
                 ):
        """ FWSA and Mamba_Block
        """
        super(SF_Block, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.drop_path = drop_path
        self.H = H
        self.W = W
        self.mlp_ratio = mlp_ratio
        self.dim = dim
        self.m1 = m1
        self.m2 = m2
        self.layers = 1
        # self.dim = 8


        self.transformer_block = TransformerBlock(dim=self.dim, num_heads=8, mlp_ratio=self.mlp_ratio)
        # self.mamba_block = MambaLayer(input_dim=self.in_channels, output_dim=self.in_channels)
        self.freq_block = DWT_block(self.in_channels, m1=self.m1, m2=self.m2, layers=self.layers)

        

        self.conv1_1 = nn.Conv2d(self.in_channels, self.in_channels*2, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.in_channels*2, self.out_channels, 1, 1, 0, bias=True)



    def forward(self, x):

        # spec_x, mamba_x = torch.split(self.conv1_1(x), (self.in_channels, self.in_channels), dim=1)# B,C,H,W
        spec_x, trans_x = torch.split(self.conv1_1(x), (self.in_channels, self.in_channels), dim=1)# B,C,H,W
        # mamba_x = self.mamba_block(mamba_x) + mamba_x
        trans_x = self.transformer_block(trans_x) + trans_x
        print("trans_x.shape:", trans_x.shape)
        spec_x = self.freq_block(spec_x) + spec_x
        print("spec_x.shape:", spec_x.shape)
        # res = self.conv1_2(torch.cat((spec_x, mamba_x), dim=1))
        res = self.conv1_2(torch.cat((spec_x, trans_x), dim=1))
        x = x + res

        return x



