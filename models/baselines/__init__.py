# pengpengzhu
# -*- coding cheer up -*- #
import torch
from .model import MemNet3

def model_generator(method, pretrained_model_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device::", device)
    
    model = None  # 初始化 model 变量
    
    if method == 'memnet3':
        model = MemNet3(in_channels=244, channels=16, num_memblock=6, num_resblock=6, drop_path=0.1, H=128, W=128).to(device)
    elif method == 'unet_attention':
        # 对于 unet_attention 方法，暂时使用 MemNet3 模型
        model = MemNet3(in_channels=244, channels=16, num_memblock=6, num_resblock=6, drop_path=0.1, H=128, W=128).to(device)
        print(f"Using MemNet3 as substitute for {method}")
    else:
        raise ValueError(f'Method {method} is not defined !!!!')
    
    # 加载预训练模型（如果路径不为空且模型已创建）
    if pretrained_model_path is not None and model is not None:
        print(f'load model from {pretrained_model_path}')  # 打印将要加载的权重路径
        try:
            checkpoint = torch.load(pretrained_model_path)    # 读取 checkpoint
            # 从 checkpoint 中取出 'state_dict'，并把多卡保存时的 'module.' 前缀去掉，再严格匹配加载到模型
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                                  strict=True)
        except Exception as e:
            print(f'Error loading pretrained model: {e}')
    
    return model