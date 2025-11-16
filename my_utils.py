from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
import hdf5storage
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
import math  # 数学函数库，主要用于计算 PSNR 等
import torch  # PyTorch 主库
from torchmetrics import StructuralSimilarityIndexMeasure


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count

    def updata(self, data):
        pass


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch,  model, optimizer):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))


class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label + 1e-5) / (label + 1e-5)

        mrae = torch.mean(error)
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.mean(sqrt_error))
        return rmse


class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=1.0):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range)
        Itrue = Itrue.reshape(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range)
        Ifake = Ifake.reshape(N, C * H * W)

        mse = nn.MSELoss(reduction='none')
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)

        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)


class Loss_RMSE_MASK(nn.Module):
    def __init__(self):
        super(Loss_RMSE_MASK, self).__init__()

    def forward(self, outputs, label, mask_3D):
        assert outputs.shape == label.shape
        c = outputs.shape[1]
        error = outputs-label
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.sum(sqrt_error) / torch.sum(mask_3D) / c)
        return rmse


class Loss_PSNR_MASK(nn.Module):
    def __init__(self):
        super(Loss_PSNR_MASK, self).__init__()

    def forward(self, im_true, im_fake, mask_3D, data_range=1.0):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range)
        Itrue = Itrue.reshape(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range)
        Ifake = Ifake.reshape(N, C * H * W)

        mse = nn.MSELoss(reduction='none')
        # err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)

        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * torch.sum(mask_3D))

        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

class Loss_MAE(nn.Module):
    def __init__(self):
        super(Loss_MAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        l1_error = torch.abs(error)
        mae = torch.mean(l1_error)
        return mae

class Loss_TV(nn.Module):


    def __init__(self, TVLoss_weight: float=1):
        super(Loss_TV, self).__init__()
        self.weight = TVLoss_weight

    def forward(self, outputs, labels):

        _, _, h, w = outputs.shape

        h_tv = torch.abs(outputs[:, :, 1:, :] - labels[:, :, :h-1, :]).mean()
        w_tv = torch.abs(outputs[:, :, :, 1:] - labels[:, :, :, :w-1]).mean()

        loss = self.weight*(h_tv + w_tv)

        return loss




class SSIMLoss_norm(nn.Module):
    def __init__(self, window_size=11, data_range=None):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=data_range,
            reduction='none'
        )
    def forward(self, output, label):
        # 输入形状: (B, C, H, W)
        print("SSIMLoss output.shape:",self.ssim(output, label).mean())
        return self.ssim(output, label).mean()  # 损失平均 SSIM

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, data_range=None):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=data_range,
            reduction='none'
        )
    def forward(self, output, label):
        # 输入形状: (B, C, H, W)
        print("SSIMLoss output.shape:",self.ssim(output, label).mean())
        return 1 - self.ssim(output, label).mean()  # 损失 = 1 - 平均 SSIM


class LocalStatLoss(nn.Module):
    def __init__(self, patch_size=8, mode='mean_var'):
        super().__init__()
        self.patch_size = patch_size
        self.mode = mode

    def _extract_patches(self, x):
        # 输入形状: (B, C, H, W) -> 输出形状: (B, C, nH, nW, pH, pW)
        patches = x.unfold(2, self.patch_size, self.patch_size) \
            .unfold(3, self.patch_size, self.patch_size)
        return patches.contiguous()

    def forward(self, pred, target):
        pred_patches = self._extract_patches(pred)  # 形状: (1,1,128,153,8,8)
        target_patches = self._extract_patches(target)

        if self.mode == 'mean_var':
            pred_mean = pred_patches.mean(dim=(4, 5))  # 计算每个块的均值
            pred_var = pred_patches.var(dim=(4, 5))  # 计算每个块的方差

            target_mean = target_patches.mean(dim=(4, 5))
            target_var = target_patches.var(dim=(4, 5))

            loss = F.mse_loss(pred_mean, target_mean) + F.mse_loss(pred_var, target_var)

        elif self.mode == 'min_max':
            pred_min = pred_patches.amin(dim=(4, 5))
            pred_max = pred_patches.amax(dim=(4, 5))

            target_min = target_patches.amin(dim=(4, 5))
            target_max = target_patches.amax(dim=(4, 5))

            loss = F.l1_loss(pred_min, target_min) + F.l1_loss(pred_max, target_max)

        return loss


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        laplacian_kernel = torch.tensor(
            [[[[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]]]], dtype=torch.float32
        )
        self.register_buffer('laplacian', laplacian_kernel)

    def forward(self, pred):
        # 计算二阶导数
        lap = F.conv2d(pred, self.laplacian, padding=1)
        return torch.mean(torch.abs(lap))


class SSIMLoss_1(nn.Module):
    def __init__(self, window_size=11, data_range=1.0, sigma=1.5):
        """
        Args:
            window_size (int): 高斯窗口大小，必须为奇数
            data_range (float): 输入数据的值范围（如 [0,1] 则为1.0）
            sigma (float): 高斯核的标准差
        """
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.sigma = sigma
        self.channels = 1  # 假设单通道输入

        # 预计算高斯核
        self._create_gaussian_kernel()

    def _create_gaussian_kernel(self):
        # 生成1D高斯核
        kernel_1d = torch.arange(self.window_size, dtype=torch.float) - (self.window_size - 1) / 2
        kernel_1d = torch.exp(-0.5 * (kernel_1d / self.sigma) ** 2)
        kernel_1d /= kernel_1d.sum()  # 归一化

        # 生成2D高斯核 (外积)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, self.window_size, self.window_size)  # 形状(1,1,H,W)
        kernel_2d = kernel_2d.repeat(self.channels, 1, 1, 1)  # 形状(C,1,H,W)

        self.register_buffer('gaussian_kernel', kernel_2d)

    def _compute_ssim_per_channel(self, x, y):
        # 输入形状: (B, C, H, W)
        C = x.size(1)
        padding = (self.window_size - 1) // 2

        # 用高斯核做卷积计算局部均值
        mu_x = F.conv2d(x, self.gaussian_kernel, padding=padding, groups=C)
        mu_y = F.conv2d(y, self.gaussian_kernel, padding=padding, groups=C)

        # 计算协方差和方差
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x*x, self.gaussian_kernel, padding=padding, groups=C) - mu_x_sq
        sigma_y_sq = F.conv2d(y*y, self.gaussian_kernel, padding=padding, groups=C) - mu_y_sq
        sigma_xy = F.conv2d(x*y, self.gaussian_kernel, padding=padding, groups=C) - mu_xy

        # 稳定常数（基于数据范围）
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        # SSIM公式
        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim_map = numerator / (denominator + 1e-6)  # 防止除零

        return ssim_map

    def forward(self, pred, target):
        # 输入形状: (B, C, H, W)
        if pred.size(1) != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {pred.size(1)}")

        ssim_map = self._compute_ssim_per_channel(pred, target)
        return 1 - ssim_map.mean()  # 损失 = 1 - 平均SSIM

    # ---------------- 工具类与评估指标（独立实现，避免依赖外部文件） ----------------

def tensor_range(x):  # 估计张量范围（用于 PSNR 动态范围判断）
        return float(x.max().item() - x.min().item() + 1e-8)  # 返回最大最小差值并加微小项防止为零

def to_uint8_img(t):  # 将模型输出或标签张量转为可保存的 8bit 图像
        t = t.detach().cpu().clamp(0, 1)  # 先裁剪到 [0,1] 并移到 CPU
        t = (t * 255.0 + 0.5).to(torch.uint8)  # 缩放至 [0,255] 并四舍五入到 uint8
        return t  # 返回 uint8 张量

def compute_rmse(pred, gt):  # 计算 RMSE 指标
        mse = F.mse_loss(pred, gt, reduction='mean')  # 先计算均方误差
        rmse = torch.sqrt(mse + 1e-12)  # 开平方得到 RMSE，并加微小项稳定
        return float(rmse.item())  # 返回 python 浮点数

def compute_psnr(pred, gt, data_range=None):  # 计算 PSNR 指标
        mse = F.mse_loss(pred, gt, reduction='mean')  # 计算均方误差
        if data_range is None:  # 如果未显式给定动态范围
            # 默认按图像范围估计动态范围（更稳健，适配归一化或 0~1 数据）
            data_range = max(tensor_range(gt), tensor_range(pred))  # 使用预测和真值的范围上界
            data_range = max(data_range, 1.0)  # 若图像接近常数，退化为 1.0（适合 0~1 归一化）
        psnr = 10.0 * math.log10((data_range ** 2) / (float(mse.item()) + 1e-12))  # 按公式计算 PSNR
        return float(psnr)  # 返回 python 浮点数

def _gaussian_window(window_size=11, sigma=1.5, device='cpu', channels=1):  # 生成二维高斯核窗口
        coords = torch.arange(window_size, device=device).float()  # 生成坐标序列
        coords -= (window_size - 1) / 2.0  # 平移到中心对称
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))  # 计算一维高斯分布
        g = g / g.sum()  # 归一化到和为 1
        window = (g[:, None] * g[None, :])  # 外积得到二维高斯核
        window = window.expand(channels, 1, window_size, window_size).contiguous()  # 扩展到每通道一个核
        return window  # 返回高斯窗口张量

def compute_ssim(img1, img2, window_size=11, sigma=1.5, data_range=1.0):  # 计算 SSIM 指标
        # 假定 img1/img2 形状为 [B,C,H,W] 且值域在 [0,1]（或传入 data_range 指定）
        C1 = (0.01 * data_range) ** 2  # SSIM 常数项 C1
        C2 = (0.03 * data_range) ** 2  # SSIM 常数项 C2
        device = img1.device  # 获取设备
        channels = img1.size(1)  # 通道数
        window = _gaussian_window(window_size, sigma, device, channels)  # 生成高斯核
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)  # 均值滤波得到 mu1
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)  # 均值滤波得到 mu2
        mu1_sq = mu1 * mu1  # mu1 的平方
        mu2_sq = mu2 * mu2  # mu2 的平方
        mu1_mu2 = mu1 * mu2  # mu1 与 mu2 的乘积
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq  # 方差估计1
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq  # 方差估计2
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2  # 协方差估计
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                    (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))  # SSIM 映射
        return float(ssim_map.mean().item())  # 返回 SSIM 平均值

def compute_sam(pred, gt, eps=1e-8):  # 计算 SAM（Spectral Angle Mapper）指标，单位为度
        # 将 [B,C,H,W] 展平到像素向量列表形式以便逐像素计算角度
        B, C, H, W = pred.shape  # 获取形状信息
        v1 = pred.permute(0, 2, 3, 1).reshape(-1, C)  # 重排为 [N, C]，N=B*H*W
        v2 = gt.permute(0, 2, 3, 1).reshape(-1, C)  # 同上用于真值
        v1 = v1.double()  # 转为 double 精度以提升稳定性
        v2 = v2.double()  # 同上
        dot = (v1 * v2).sum(dim=1)  # 计算逐像素向量点积
        n1 = v1.norm(dim=1)  # 计算逐像素向量范数1
        n2 = v2.norm(dim=1)  # 计算逐像素向量范数2
        cos_theta = dot / (n1 * n2 + eps)  # 计算夹角余弦并加微小项
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 将余弦值裁剪到 [-1,1]
        angle_rad = torch.acos(cos_theta)  # 反余弦得到弧度
        angle_deg = angle_rad * (180.0 / math.pi)  # 弧度转角度
        return float(angle_deg.mean().item())  # 返回平均 SAM（度）

class Loss_MRAE_RMSE_SSIM(nn.Module):
    def __init__(self, mrae_weight=0.2, rmse_weight=0.7, ssim_weight=0.1):
        super(Loss_MRAE_RMSE_SSIM, self).__init__()
        
        # 初始化各个单独的损失函数
        self.loss_mrae = Loss_MRAE()
        self.loss_rmse = Loss_RMSE()
        self.loss_ssim = SSIMLoss() #is 1-ssim
        
        # 设定每个损失函数的权重（超参数）
        self.mrae_weight = mrae_weight
        self.rmse_weight = rmse_weight
        self.ssim_weight = ssim_weight

    def forward(self, outputs, labels):
        # 计算 MRAE 损失
        mrae_loss = self.loss_mrae(outputs, labels)
        
        # 计算 RMSE 损失
        rmse_loss = self.loss_rmse(outputs, labels)

        
        # 计算 SSIM 损失
        ssim_loss = self.loss_ssim(outputs, labels)
        print("mrae_loss:",mrae_loss.item(),"rmse_loss:",rmse_loss.item(),"ssim_loss:",ssim_loss.item())

        # 计算总的混合损失，按照权重进行加权
        total_loss = (self.mrae_weight * mrae_loss) + (self.rmse_weight * rmse_loss) + (self.ssim_weight * ssim_loss)
        
        return total_loss


class Loss_TV_RMSE_SSIM(nn.Module):
    def __init__(self, tv_weight=1.0, rmse_weight=1.0, ssim_weight=1.0):
        super(Loss_TV_RMSE_SSIM, self).__init__()

        # 初始化各个单独的损失函数
        self.loss_tv = Loss_TV()  # 替换 MRAE 为 TV Loss
        self.loss_rmse = Loss_RMSE()
        self.loss_ssim = SSIMLoss()

        # 设定每个损失函数的权重（超参数）
        self.tv_weight = tv_weight
        self.rmse_weight = rmse_weight
        self.ssim_weight = ssim_weight

    def forward(self, outputs, labels):
        # 计算 TV 损失
        tv_loss = self.loss_tv(outputs, labels)

        # 计算 RMSE 损失
        rmse_loss = self.loss_rmse(outputs, labels)

        # 计算 SSIM 损失
        ssim_loss = self.loss_ssim(outputs, labels)
        print("tv_loss:", tv_loss.item(), "rmse_loss:", rmse_loss.item(), "ssim_loss:", ssim_loss.item())

        # 计算总的混合损失，按照权重进行加权
        total_loss = (self.tv_weight * tv_loss) + (self.rmse_weight * rmse_loss) + (self.ssim_weight * ssim_loss)

        return total_loss

# if __name__ == "__main__":
#     # 测试代码
#     batch_size = 2
#     channels = 244
#     height = 244
#     width = 244

#     # 创建随机的输入和标签（模拟数据）
#     outputs = torch.rand((batch_size, channels, height, width))  # 模拟模型输出
#     print("outputs.shape:",outputs.shape,type(outputs),outputs.max())
#     labels = torch.rand((batch_size, channels, height, width))   # 模拟真实标签

#     # 初始化损失函数
#     loss_fn = Loss_TV_RMSE_SSIM(tv_weight=0.2, rmse_weight=0.7, ssim_weight=0.1)

#     # 计算损失
#     print("outputs.shape:",outputs.shape,type(outputs))  
#     loss = loss_fn(outputs, labels)
#     print("loss.shape:",loss ,loss.shape,type(loss))
#     loss = loss_fn(outputs, labels).mean()
#     print("loss.shape:",loss ,loss.shape,type(loss))
#     print(f"Total Loss: {loss.item()}")

