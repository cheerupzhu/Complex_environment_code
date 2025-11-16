# pengpengzhu  # 文件作者/标记（可选）
# -*- coding cheer up -*- #  # 自定义注释，无实际语义；不会影响 Python 解释器
import numpy as np  # 导入 NumPy，用于数值计算和数组操作
import matplotlib.pyplot as plt
import h5py
import os
def wavelengths_61(start=400, stop=1000, n=61):  # 生成线性等间隔的波长序列（单位：nm）
    return np.linspace(start, stop, n)  # 在[start, stop]区间内等间隔采样n个波长

def band_avg(cube, wls, center_nm, halfwidth_nm=10):  # 在中心波长±半宽范围内做带通平均
    lo, hi = center_nm - halfwidth_nm, center_nm + halfwidth_nm  # 计算带宽下限与上限
    sel = np.where((wls >= lo) & (wls <= hi))[0]  # 找到落在带宽范围内的波段索引
    if sel.size == 0:  # 边缘情况：没有波段命中该窗口
        sel = np.array([np.argmin(np.abs(wls - center_nm))])  # 退化为与中心最近的单一波段
    return cube[sel].mean(axis=0)  # 对选中波段在光谱维求均值，输出(H, W)

def percentile_normalize(img, p_low=1, p_high=99):  # 用百分位拉伸到[0,1]，抑制极值
    lo, hi = np.percentile(img, p_low), np.percentile(img, p_high)  # 取低/高百分位作为映射上下限
    if hi <= lo: hi = lo + 1e-6  # 防止分母为0：若上下限过近则微调上限
    x = np.clip((img - lo) / (hi - lo), 0, 1)  # 线性归一化并裁剪到[0,1]
    return x  # 返回标准化后的图像

def gamma_encode(x, gamma=2.2):  # 伽马编码（显示设备友好）
    return np.clip(x, 0, 1) ** (1/gamma)  # 先裁剪到[0,1]再做反伽马变换

def pseudo_rgb_single_polar(data,  # 将单偏振光谱立方映射成伪彩色RGB
                            angle_deg=0,           # 选择偏振角：0/45/90/135之一
                            wl_B=460, wl_G=550, wl_R=650,  # 蓝/绿/红通道中心波长（nm）
                            halfwidth_nm=10,  # 每个颜色通道的半带宽（±halfwidth_nm）
                            p_low=1, p_high=99, gamma=2.2):  # 百分位拉伸与伽马参数
    """
    data: shape (4, 61, H, W) 或 (61, H, W)  # 输入可以包含偏振维或不包含
    返回: uint8 RGB, shape (H, W, 3)          # 输出8位RGB图像
    """
    # 取出单偏振光谱立方
    if data.ndim == 4:  # 若输入包含偏振维度
        angles = {0:0, 45:1, 90:2, 135:3}  # 偏振角到索引的映射表
        idx = angles.get(angle_deg, 0)  # 获取所选偏振角对应的索引，默认0度
        # print("idx shape:",idx.)
        cube = data[idx]            # 取出该偏振通道的光谱立方，形状(61, H, W)
        print("cube shape: ", cube.shape)
    elif data.ndim == 3:  # 若输入已为单偏振/无偏振维
        cube = data  # 直接使用
    else:
        raise ValueError("data 形状应为 (4,61,H,W) 或 (61,H,W)")  # 输入维度不符合预期则报错

    n_wl, H, W = cube.shape  # 解析光谱数与空间尺寸
    wls = wavelengths_61(400, 1000, n_wl)  # 依据光谱数生成对应波长刻度（默认400–1000 nm）

    # 选取B/G/R三个波段并做带通平均
    B = band_avg(cube, wls, wl_B, halfwidth_nm)  # 蓝通道带通平均 -> (H, W)
    G = band_avg(cube, wls, wl_G, halfwidth_nm)  # 绿通道带通平均 -> (H, W)
    R = band_avg(cube, wls, wl_R, halfwidth_nm)  # 红通道带通平均 -> (H, W)

    # 每个通道独立做百分位拉伸与伽马校正
    Rn = gamma_encode(percentile_normalize(R, p_low, p_high), gamma)  # 红通道处理
    Gn = gamma_encode(percentile_normalize(G, p_low, p_high), gamma)  # 绿通道处理
    Bn = gamma_encode(percentile_normalize(B, p_low, p_high), gamma)  # 蓝通道处理

    rgb = np.stack([Rn, Gn, Bn], axis=-1)  # 按(R,G,B)堆叠成(H, W, 3)浮点图（0~1）
    return (rgb * 255 + 0.5).astype(np.uint8)  # 缩放到[0,255]并四舍五入，转换为uint8

# # 用法示例：
# path = r"E:\quwu\Unet_Dehazing-main\dataset\PolHSI\building\clear\HSI_R_Image_20250812151024272_pol_all.h5"
# with h5py.File(path, "r") as f:
#     print("可用数据集(keys):", list(f.keys()))
#     # 兼容你截图里常见的命名
#     key_candidates = ["hsi_R", "hsi_Hazy", "HSI_R", "HSR_R"]
#     key = next((k for k in key_candidates if k in f.keys()), None)
#     if key is None:
#         raise KeyError("文件内未找到 hsi_R/hsr_R 等数据集，请打印 keys 确认真实名称。")
#     hsi_R = f[key][...]  # 读取成 numpy 数组（形状应为 (4,61,H,W)）

npy_path = r"/data/ppm/dehazing_PPM/Test_Results/lunwen_0001/npy/sample_000000.npy"
# 读取文件（兼容 .npz 和 .npy（字典或数组））
data_loaded = np.load(npy_path, allow_pickle=True).item()  # 读取 .npy 文件内容
print("data_loaded type:", type(data_loaded))  # 打印加载对象类型，调试用
# 目标变量名：尽量匹配你后续要用的 hsi_R 名称
hsi_R = None

hsi_R = data_loaded["hazy"]
print("hsi_R shape:", hsi_R.shape, "dtype:", hsi_R.dtype)


rgb = pseudo_rgb_single_polar(hsi_R, angle_deg=0, wl_B=460, wl_G=550, wl_R=650, halfwidth_nm=10)  # 假设hsi_R形状为(4,61,H,W)或(61,H,W)

# 注意：以下显示需先 `import matplotlib.pyplot as plt`
plt.imshow(rgb)  # 显示伪彩色图像
plt.axis("off")   # 关闭坐标轴显示
plt.show()  # 刷新并弹出绘图窗口

# 假设 rgb 是浮点数 [0,1]
if rgb.dtype != np.uint8:
    rgb_uint8 = np.uint8(np.clip(rgb * 255, 0, 255))
else:
    rgb_uint8 = rgb

H, W = rgb_uint8.shape[:2]
dpi = 300
figsize = (W / dpi, H / dpi)

fig = plt.figure(figsize=figsize, dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(rgb_uint8, interpolation='nearest')
ax.axis('off')
fig.patch.set_alpha(0.0)  # 背景透明

out_path = "/data/ppm/dehazing_PPM/band_vis_hazy/sample0/pseudo_rgb_hazy_0.png"  # 保存路径
fig.savefig(out_path, dpi=dpi, transparent=True, pad_inches=0)
plt.close(fig)
