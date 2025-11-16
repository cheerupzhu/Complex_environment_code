# -*- coding: utf-8 -*-  # 指定文件编码为 UTF-8，确保中文注释正常显示
import os  # 操作系统相关功能，如路径、环境变量、文件夹创建
import math  # 数学函数库，主要用于计算 PSNR 等
import argparse  # 命令行参数解析
import warnings  # 警告过滤
warnings.filterwarnings("ignore", message="Failed to load image Python extension")  # 忽略特定警告信息，保持输出整洁
import h5py  # 处理 HDF5 文件格式
import torch  # PyTorch 主库
import numpy as np
import torch.nn.functional as F  # 常用函数（如卷积、插值、激活等）
from torch.utils.data import DataLoader  # 数据加载器
from tqdm import tqdm  # 进度条显示
from my_utils import AverageMeter , compute_rmse, compute_psnr ,compute_sam ,compute_ssim

# ---- 根据你的项目结构导入数据集与模型构造器（保持与 train.py 一致） ----
from data_load2 import H5HazyClearTestDataset as H5HazyClearDataset  # 导入你在训练脚本中使用的数据集类
from models import *


# ---------------- 工具类与评估指标（独立实现，避免依赖外部文件） ----------------

# def save_image_grid(tensor_chw, save_path):  # 保存单张图像（C,H,W）到指定路径（PNG）
#     import torchvision.utils as vutils  # 延迟导入 torchvision.utils（避免不必要依赖）
#     vutils.save_image(tensor_chw.clamp(0, 1), save_path)  # 保存为 PNG，自动按通道处理

# ------------------------------ 主评测流程 ------------------------------

def parse_args():  # 定义并解析命令行参数
    parser = argparse.ArgumentParser(description="PHI 传感器模型测试脚本（推理与定量评测）")
    parser.add_argument("--method", type=str, default="unet_attention", help="模型名称，需与训练时一致")
    parser.add_argument("--gpu_id", type=str, default="1", help="选择使用的 GPU 编号")
    parser.add_argument("--pretrained_model_path", type=str, default="/data/ppm/dehazing_PPM/Save_model_fusion/unet_attention/000001_true/net_14299epoch.pth", help="已训练权重文件路径 .pth")
    parser.add_argument("--test_data_dir", type=str, default='/data/ppm/dehazing_PPM/PPM_hazy_data1/Test_true_hazy', help="测试数据集根目录")
    parser.add_argument("--batch_size", type=int, default=1, help="测试批大小（通常为1）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 的工作进程数")
    parser.add_argument("--save_dir", type=str, default="./Test_Results/lunwen_hazy_00002", help="指标与可视化结果的保存目录")
    args = parser.parse_args()
    return args

def setup_env(args):  # 根据参数设置运行环境（如 GPU）
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 固定 CUDA 设备顺序为 PCI_BUS_ID
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # 指定可见 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备：优先使用 GPU
    return device  # 返回设备对象

def build_loader(args):
    dataset = H5HazyClearDataset(root_dir=args.test_data_dir)  # 这里不再需要 patch_size/arg
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    return dataset, loader
def load_model(args, device):  # 创建并加载模型
    model = model_generator(args.method, args.pretrained_model_path)  # 使用与训练一致的 model_generator 加载结构和权重
    model = model.to(device)  # 将模型移动到设备（GPU/CPU）
    model.eval()  # 设为评测模式，关闭 BN/Dropout 的训练行为
    return model  # 返回模型

def evaluate_once(model, batch, device, save_dir=None,  start_idx=0):  # 对一个批次数据做推理并计算指标
    inp = batch["hazy"].to(device)  # 取输入（带雾/需重建的观测），移动到设备
    gt  = batch["clear"].to(device)  # 取对应的清晰真值，移动到设备
    with torch.no_grad():  # 关闭梯度计算，加速并节省显存
        pred = model(inp)  # 前向推理得到重建结果
        pred = torch.clamp(pred, 0.0, 1.0)  # 将输出裁剪到 [0,1]，保证指标计算稳定

    # # === 关键：指标全部挪到 CPU 计算，避免在 GPU 上产生 1GB 级中间张量 ===
    # pred_cpu = pred.detach().float().cpu()
    # gt_cpu   = gt.detach().float().cpu()
    # rmse = compute_rmse(pred_cpu, gt_cpu)
    # psnr = compute_psnr(pred_cpu, gt_cpu, data_range=1.0)
    # ssim = compute_ssim(pred_cpu, gt_cpu, window_size=11, sigma=1.5, data_range=1.0)
    # sam  = compute_sam(pred_cpu, gt_cpu)
    # 计算指标：RMSE、PSNR、SSIM、SAM（值均为越大越好，SAM 为角度通常越小越好，这里也统计其平均值）
    rmse = compute_rmse(pred, gt)  # 计算 RMSE
    psnr = compute_psnr(pred, gt, data_range=1.0)  # 计算 PSNR（假定值域 0~1）
    ssim = compute_ssim(pred, gt, window_size=11, sigma=1.5, data_range=1.0)  # 计算 SSIM
    sam  = compute_sam(pred, gt)  # 计算 SAM（度）

    npy_dir = os.path.join(save_dir, "npy")  # 保存 .npy 的子目录
    os.makedirs(npy_dir, exist_ok=True)  # 如不存在则创建
    B = pred.size(0)  # 获取批大小
    for b in range(B):  # 遍历批内样本
        base_idx = start_idx + b  # 全局编号
        # npy_path = os.path.join(npy_dir, f"sample_{base_idx:06d}.npy")
        npy_path = os.path.join(npy_dir, f"sample_{base_idx:06d}.h5")
        # 转 numpy
        hazy_np = inp[b].detach().cpu().numpy().astype(np.float32)  # (244,H,W) 输入
        pred_np = pred[b].detach().cpu().numpy().astype(np.float32)  # (244,H,W) 输出
        clear_np = gt[b].detach().cpu().numpy().astype(np.float32)  # (244,H,W) 真值
        def to_4_61_HW(arr):
            # arr: (244, H, W) -> (4, 61, H, W)
            assert arr.ndim == 3, "输入数组不是 (244,H,W) 形状"
            c, H, W = arr.shape
            assert c == 244, f"第一维应该是 244，但得到 {c}"
            return arr.reshape(4, 61, H, W)
        
        hazy_np = to_4_61_HW(hazy_np)
        pred_np = to_4_61_HW(pred_np)
        clear_np = to_4_61_HW(clear_np)
    

        # 保存为 dict，一次性写入 .npy
        # np.save(npy_path, {
        #     "hazy": hazy_np, # (4,61,H,W)
        #     "pred": pred_np, # (4,61,H,W)
        #     "clear": clear_np, # (4,61,H,W)
        # })
        arr = hazy_np.astype(np.float32, copy=False)  #  (4,61,H,W)
        _, _, H, W = arr.shape
        with h5py.File(npy_path, 'w') as f:
            dset = f.create_dataset(
                'hsi_R',
                data=arr,
                compression='gzip',  # 开启压缩
                compression_opts=4,  # 压缩等级 0-9，越大越小但更耗时
                shuffle=True,  # 打乱过滤，利于压缩
                chunks=(1, 16, H, W)  # 合理分块，便于按块读写
            )

    # # 可选保存重建图像（批内逐张保存）
    # if save_images and save_dir is not None:  # 若需要保存图像且给出保存目录
    #     os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)  # 创建图像保存子目录
    #     B = pred.size(0)  # 获取批大小
    #     for b in range(B):  # 遍历批内样本
    #         # 文件名采用全局计数索引，避免重名
    #         fname = os.path.join(save_dir, "images", f"pred_{start_idx + b:06d}.png")  # 构造文件路径
    #         save_image_grid(pred[b].cpu(), fname)  # 保存单张预测图像

    return rmse, psnr, ssim, sam  # 返回四个指标

def main():  # 主函数入口
    args = parse_args()  # 解析命令行参数
    device = setup_env(args)  # 配置运行环境并得到设备

    # 构建数据与模型
    dataset, loader = build_loader(args)  # 创建测试数据加载器
    model = load_model(args, device)  # 创建并加载模型

    # 指标累计器
    rmse_meter = AverageMeter()  # RMSE 平均器
    psnr_meter = AverageMeter()  # PSNR 平均器
    ssim_meter = AverageMeter()  # SSIM 平均器
    sam_meter  = AverageMeter()  # SAM 平均器

    # 准备保存目录与逐样本 CSV
    os.makedirs(args.save_dir, exist_ok=True)  # 创建结果保存目录
    csv_path = os.path.join(args.save_dir, "per_image_metrics.csv")  # 逐图像指标的 CSV 路径
    all_csv_lines = ["index,rmse,psnr,ssim,sam\n"]  # CSV 文件表头

    # 遍历测试集并评测
    idx_base = 0  # 用于命名与索引的起始编号
    pbar = tqdm(loader, desc="Testing", total=len(loader))  # 创建进度条
    for i, batch in enumerate(pbar):  # 逐批遍历数据
        rmse, psnr, ssim, sam = evaluate_once(  # 对当前批做推理与评测
            model, batch, device,
            save_dir=args.save_dir,
            start_idx=idx_base
        )
        bs = batch["hazy"].size(0)  # 当前批大小
        rmse_meter.update(rmse, bs)  # 更新 RMSE 平均器
        psnr_meter.update(psnr, bs)  # 更新 PSNR 平均器
        ssim_meter.update(ssim, bs)  # 更新 SSIM 平均器
        sam_meter.update(sam, bs)    # 更新 SAM 平均器

        # 写入逐样本 CSV（按批视为同指标，若需严格到每张，可在 evaluate_once 内返回逐张指标）
        for b in range(bs):  # 遍历批内样本
            all_csv_lines.append(f"{idx_base + b},{rmse:.6f},{psnr:.6f},{ssim:.6f},{sam:.6f}\n")  # 记录一行 CSV
        idx_base += bs  # 更新全局索引

        pbar.set_postfix({  # 在进度条尾部显示当前平均指标
            "RMSE": f"{rmse_meter.avg:.4f}",
            "PSNR": f"{psnr_meter.avg:.2f}",
            "SSIM": f"{ssim_meter.avg:.4f}",
            "SAM(deg)": f"{sam_meter.avg:.3f}"
        })

    # 写出逐样本指标 CSV
    with open(csv_path, "w", encoding="utf-8") as f:  # 打开 CSV 文件
        f.writelines(all_csv_lines)  # 写入全部行

    # 汇总指标也保存到一个简要文件
    summary_path = os.path.join(args.save_dir, "summary.txt")  # 汇总文件路径
    with open(summary_path, "w", encoding="utf-8") as f:  # 打开汇总文件
        f.write(f"RMSE: {rmse_meter.avg:.6f}\n")  # 写入 RMSE 平均
        f.write(f"PSNR: {psnr_meter.avg:.6f}\n")  # 写入 PSNR 平均
        f.write(f"SSIM: {ssim_meter.avg:.6f}\n")  # 写入 SSIM 平均
        f.write(f"SAM(deg): {sam_meter.avg:.6f}\n")  # 写入 SAM 平均（度）

    # 同时在控制台打印最终指标
    print("=" * 60)  # 打印分隔线
    print(f"Test Finished - N={len(dataset)}")  # 打印测试样本数
    print(f"RMSE: {rmse_meter.avg:.6f}")  # 打印 RMSE 平均
    print(f"PSNR: {psnr_meter.avg:.6f}")  # 打印 PSNR 平均
    print(f"SSIM: {ssim_meter.avg:.6f}")  # 打印 SSIM 平均
    print(f"SAM(deg): {sam_meter.avg:.6f}")  # 打印 SAM 平均（度）
    print(f"Per-image CSV: {csv_path}")  # 打印逐图像 CSV 路径
    print(f"Summary: {summary_path}")  # 打印汇总文件路径
    if args.save_images:  # 若保存了图像
        print(f"Images saved to: {os.path.join(args.save_dir, 'images')}")  # 打印图像保存目录

if __name__ == "__main__":  # 脚本入口
    torch.set_grad_enabled(False)
    main()  # 调用主函数