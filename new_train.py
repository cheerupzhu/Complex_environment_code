# pengpengzhu
# -*- coding cheer up -*- #
# -*- coding: utf-8 -*-  # 指定源码编码为 UTF-8
import os  # 操作系统相关（路径、环境变量等）
import warnings  # 过滤警告
warnings.filterwarnings("ignore", message="Failed to load image Python extension")  # 忽略 torchvision 图像扩展的加载警告
import time  # 计时
import argparse  # 解析命令行参数

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数（本脚本里未强依赖）
from torch.backends import cudnn  # cuDNN 加速相关
from torch.utils.data import DataLoader  # 数据加载器
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练（前向半精度、缩放梯度）
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 记录
from tqdm import tqdm  # 进度条

# ===== 你的项目内模块（请确保这些路径/名字与你工程一致） =====
from data_load2 import H5HazyClearDataset  # 你前面写的数据集：返回 {"hazy": (B, C, H, W), "clear": (B, C, H, W)}
# from Model.CreateModel import model_generator  # 模型工厂函数：根据 method 创建并可加载预训练
from models import *



from my_utils import Loss_RMSE, Loss_PSNR, Loss_MAE, AverageMeter, save_checkpoint, initialize_logger  # 常用工具与损失

def parse_args():
    """解析命令行参数（统一入口）"""
    parser = argparse.ArgumentParser(description="Fusion Training Script (AMP + CosineLR per-batch)")  # 创建解析器并附描述
    parser.add_argument("--method", type=str, default="unet_attention", help="模型名（传给 model_generator）")  # 模型标识
    parser.add_argument("--epochs", type=int, default=15000, help="训练轮数")  # 训练总轮数
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")  # batch size
    parser.add_argument("--init_lr", type=float, default=4e-4, help="初始学习率")  # 初始学习率
    parser.add_argument("--weight_path", type=str, default="./Save_model_fusion/unet_attention/000001_zuihou", help="权重保存目录")  # 权重输出目录
    parser.add_argument("--pretrained_model_path", type=str, help="预训练权重路径（可为空）")  # 预训练路径
    parser.add_argument("--train_data_dir", type=str, default="/data/ppm/quwudaima/out_images", help="训练数据根目录")  # 数据根目录
    parser.add_argument("--train_patch_size", type=int, nargs="+", default=[512, 512], help="训练裁剪 patch 大小（512 或 512 512）")  # patch 大小
    parser.add_argument("--val_split", type=float, default=0, help="验证集比例")  # 验证比例
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 线程数（Windows 先用 0）")  # 加载线程
    parser.add_argument("--amp", action="store_true", help="启用 AMP 混合精度训练")  # 是否启用 AMP
    parser.add_argument("--grad_clip", type=float, default=0.0, help="梯度裁剪阈值（0 表示不裁剪）")  # 梯度裁剪
    parser.add_argument("--gpu_id", type=str, default="2", help="CUDA_VISIBLE_DEVICES（多卡时指定可见卡）")  # 可见 GPU
    parser.add_argument("--log_dir", type=str, default="", help="TensorBoard 日志目录（默认用 weight_path）")  # 日志目录
    args = parser.parse_args()  # 解析参数
    return args  # 返回解析结果

def normalize_patch_size(ps):
    """把 patch_size 规范成 (h, w) 元组"""
    if isinstance(ps, int):  # 如果是单个 int（比如 512）
        return (ps, ps)  # 返回 (512, 512)
    if isinstance(ps, (list, tuple)):  # 如果是列表或元组
        assert len(ps) in (1, 2), "train_patch_size 只允许一个或两个数"  # 校验长度
        return (ps[0], ps[0]) if len(ps) == 1 else (ps[0], ps[1])  # 单数 -> 方形；两个数 -> 高宽
    raise ValueError("train_patch_size 类型不正确，应为 int / list / tuple")  # 其它类型抛错

def build_dataloaders(opt, device):
    """构建 Dataset 与 DataLoader"""
    patch_size = normalize_patch_size(opt.train_patch_size)  # 规范化 patch 大小
    dataset = H5HazyClearDataset(root_dir=opt.train_data_dir, patch_size=patch_size, arg=True)  # 构建训练全集（内部会做随机裁剪）
    print("数据对总数:", len(dataset))  # 打印数据集大小
    # 划分训练/验证（按比例随机划分）
    train_size = int((1.0 - opt.val_split) * len(dataset))  # 训练集大小
    val_size = len(dataset) - train_size  # 验证集大小
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])  # 随机切分
    print("训练集:", len(train_dataset), "验证集:", len(val_dataset))  # 打印切分结果
    # DataLoader 参数：pin_memory 提升 H2D 拷贝速度；persistent_workers True 需要 num_workers>0
    persistent = True if opt.num_workers > 0 else False  # 是否持久化工作线程
    train_loader = DataLoader(  # 构建训练集 DataLoader
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
        pin_memory=True, persistent_workers=persistent, drop_last=False
    )
    val_loader = DataLoader(  # 构建验证集 DataLoader
        val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
        pin_memory=True, persistent_workers=persistent, drop_last=False
    )
    # return train_loader, val_loader  # 返回两个加载器
    return train_loader, train_loader

def build_model_optimizer_scheduler(opt, device, steps_per_epoch):
    """构建模型、优化器、学习率调度器、损失函数等"""
    # 构建模型（你的 model_generator 应该能根据 method 返回已构建好的模型；若内部 .cuda() 了也没关系，这里再 to(device) 一次）
    model = model_generator(opt.method, opt.pretrained_model_path)  # 根据方法名与预训练路径创建模型
    model = model.to(device)  # 模型放到设备（CPU/GPU）
    # 优化器（Adam）
    optimizer = torch.optim.Adam(  # 使用 Adam 优化器
        filter(lambda p: p.requires_grad, model.parameters()),  # 仅优化需要梯度的参数
        lr=opt.init_lr, betas=(0.9, 0.999)  # 学习率及 betas
    )
    # 学习率调度器（按 batch 调用 step，因此 T_max=epochs * steps_per_epoch）
    total_steps = max(1, opt.epochs * max(1, steps_per_epoch))  # 总步数（至少为 1，防除零）
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  # 余弦退火学习率
        optimizer, T_max=total_steps, eta_min=1e-6  # 最低学习率 1e-6，可按需调整
    )
    # 损失函数：训练用 RMSE，验证记录 PSNR（与原脚本保持一致）
    criterion_rmse = Loss_RMSE()  # 创建 RMSE 损失
    criterion_psnr = Loss_PSNR()  # 创建 PSNR 指标（做验证汇报）
    return model, optimizer, lr_scheduler, criterion_rmse, criterion_psnr  # 返回构建结果

def train_one_epoch(epoch, opt, model, optimizer, lr_scheduler, scaler, criterion_rmse, train_loader, device, writer):
    """单个 epoch 的训练过程"""
    model.train()  # 切换到训练模式（启用 BN/Dropout 等）
    losses = AverageMeter()  # 平均器，用于统计当前 epoch 的平均损失（样本加权）
    pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch {epoch+1}/{opt.epochs}")  # 进度条包装迭代器
    for step, sample in pbar:  # 遍历每个 mini-batch
        inputs = sample["hazy"].to(device, non_blocking=True)  # 取输入并拷到设备（non_blocking 需配合 pin_memory）
        targets = sample["clear"].to(device, non_blocking=True)  # 取 GT 并拷到设备

        # inputs = Add_hazy(targets, )



        optimizer.zero_grad(set_to_none=True)  # 梯度清零（set_to_none 更省内存）
        with autocast(enabled=opt.amp):  # 开启/关闭混合精度（只影响前向与损失计算）
            outputs = model(inputs)  # 前向计算得到预测
            loss = criterion_rmse(outputs, targets)  # 计算 RMSE 损失（你的实现已是 mean，不用再除以 batch）
        if opt.amp:  # 若启用 AMP
            scaler.scale(loss).backward()  # 缩放后反向传播以避免溢出
            if opt.grad_clip > 0:  # 若设置了梯度裁剪
                scaler.unscale_(optimizer)  # 先反缩放到真实梯度
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip)  # 裁剪梯度范数
            scaler.step(optimizer)  # scaler 触发优化器步进
            scaler.update()  # 动态调整缩放因子
        else:  # 未启用 AMP 的普通路径
            loss.backward()  # 反向传播
            if opt.grad_clip > 0:  # 可选梯度裁剪
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip)  # 裁剪
            optimizer.step()  # 优化器步进
        lr_scheduler.step()  # 每个 batch 更新一次学习率（T_max 需等于总步数）
        losses.update(loss.item(), inputs.size(0))  # 用本 batch 的样本数加权更新平均损失
        cur_lr = optimizer.param_groups[0]["lr"]  # 读取当前学习率（第一个 param group）
        pbar.set_postfix(loss=f"{losses.avg:.4f}", lr=f"{cur_lr:.2e}")  # 在进度条尾部显示平均损失与学习率
    # 写入 TensorBoard（训练损失）
    if writer is not None:  # 若启用了 TB 记录
        writer.add_scalar("Loss/Train_RMSE", losses.avg, epoch)  # 记录本 epoch 的训练平均 RMSE
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)  # 记录学习率
    return losses.avg  # 返回训练平均损失

def validate_one_epoch(epoch, opt, model, criterion_rmse, criterion_psnr, val_loader, device, writer):
    """单个 epoch 的验证过程（不求梯度）"""
    model.eval()  # 切换到评估模式（关闭 BN 统计更新/Dropout）
    val_rmse = AverageMeter()  # 验证 RMSE 平均器
    val_psnr = AverageMeter()  # 验证 PSNR 平均器
    with torch.no_grad():  # 关闭梯度，显存/速度更好
        for sample in tqdm(val_loader, total=len(val_loader), desc="Validate", leave=False):  # 验证进度条
            inputs = sample["hazy"].to(device, non_blocking=True)  # 验证输入
            targets = sample["clear"].to(device, non_blocking=True)  # 验证 GT
            outputs = model(inputs)  # 前向
            rmse = criterion_rmse(outputs, targets).item()  # 计算 RMSE 数值
            psnr = criterion_psnr(outputs, targets).item()  # 计算 PSNR 数值
            val_rmse.update(rmse, inputs.size(0))  # 按样本加权更新 RMSE
            val_psnr.update(psnr, inputs.size(0))  # 按样本加权更新 PSNR
    # 写入 TensorBoard（验证损失/指标）
    if writer is not None:  # 若启用了 TB
        writer.add_scalar("Loss/Val_RMSE", val_rmse.avg, epoch)  # 记录验证 RMSE
        writer.add_scalar("Metric/Val_PSNR", val_psnr.avg, epoch)  # 记录验证 PSNR
    return val_rmse.avg, val_psnr.avg  # 返回验证平均 RMSE 与 PSNR

def main():
    """主函数：整合数据、模型、训练与验证"""
    opt = parse_args()  # 解析命令行参数
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 设定 CUDA 设备排序（按 PCI 顺序）
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id  # 控制可见 GPU（如 "0" 或 "0,1"）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备（GPU 优先）
    print("device:", device)  # 打印当前设备
    cudnn.benchmark = True  # 启用 cuDNN benchmark（对固定输入尺寸可加速）
    # 准备日志与权重目录
    os.makedirs(opt.weight_path, exist_ok=True)  # 创建权重输出目录
    tb_dir = opt.log_dir if opt.log_dir else opt.weight_path  # 若未指定日志目录，就用权重目录
    writer = SummaryWriter(log_dir=tb_dir)  # 创建 TensorBoard 记录器
    logger = initialize_logger(os.path.join(opt.weight_path, "train.log"))  # 初始化文本日志记录
    # 构建数据加载器
    train_loader, val_loader = build_dataloaders(opt, device)  # 构建训练与验证 DataLoader
    # 构建模型、优化器、调度器与损失
    model, optimizer, lr_scheduler, criterion_rmse, criterion_psnr = build_model_optimizer_scheduler(
        opt, device, steps_per_epoch=len(train_loader)
    )  # 传入每个 epoch 的步数以设定 T_max
    # 准备 AMP 缩放器
    scaler = GradScaler(enabled=opt.amp)  # 根据 --amp 开关启用/禁用混合精度
    # 训练循环全局变量
    best_val_rmse = float("inf")  # 历史最优验证 RMSE（越小越好）
    start_time = time.time()  # 记录起始时间
    # 进入多轮训练
    for epoch in range(opt.epochs):  # 遍历每个 epoch
        print(f"\n=== 第 {epoch+1}/{opt.epochs} 轮训练开始 ===")  # 打印轮次信息
        train_rmse = train_one_epoch(epoch, opt, model, optimizer, lr_scheduler, scaler, criterion_rmse, train_loader, device, writer)  # 训练一个 epoch
        val_rmse, val_psnr = validate_one_epoch(epoch, opt, model, criterion_rmse, criterion_psnr, val_loader, device, writer)  # 验证一个 epoch
        # 统计耗时并输出日志
        epoch_time = time.time() - start_time  # 计算本轮耗时
        start_time = time.time()  # 重置起始时间
        cur_lr = optimizer.param_groups[0]["lr"]  # 当前学习率
        print("Epoch[%04d]  Time: %.1fs  LR: %.2e  Train_RMSE: %.4f  Val_RMSE: %.4f  Val_PSNR: %.2f" %
              (epoch+1, epoch_time, cur_lr, train_rmse, val_rmse, val_psnr))  # 控制台输出结果
        logger.info("Epoch[%04d]  Time: %.1fs  LR: %.2e  Train_RMSE: %.4f  Val_RMSE: %.4f  Val_PSNR: %.2f" %
                    (epoch+1, epoch_time, cur_lr, train_rmse, val_rmse, val_psnr))  # 写入日志文件
        # 保存模型（更优或最后一轮）
        if (val_rmse < best_val_rmse) or (epoch == opt.epochs - 1):  # 如果取得更好结果或到了最后一轮
            print(f"保存权重到: {opt.weight_path}")  # 打印保存路径
            save_checkpoint(opt.weight_path, epoch, model, optimizer)  # 使用你项目里的保存函数保存 checkpoint
            best_val_rmse = min(best_val_rmse, val_rmse)  # 更新 best RMSE
    writer.close()  # 关闭 TensorBoard 记录器

if __name__ == "__main__":
    main()  # 入口：执行主函数
