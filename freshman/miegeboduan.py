# pengpengzhu                                                     # 文件作者标识（可选）
# -*- coding cheer up -*- #                                       # 自定义注释，无实际语义

import os                                                         # 操作系统相关（路径、目录、创建文件夹等）
import h5py                                                       # 读取 .h5 数据文件
import numpy as np                                                # 数值计算库
import matplotlib.pyplot as plt                                   # 可视化绘图库

# ===== 1) 工具函数 =====
def percentile_normalize(x, p_low=1, p_high=99, eps=1e-8):       # 按分位数做归一化，抑制极端值影响
    """按分位数归一化到 [0,1]，更鲁棒"""                         # 函数文档字符串
    lo = np.percentile(x, p_low)                                  # 计算下分位阈值
    hi = np.percentile(x, p_high)                                 # 计算上分位阈值
    if hi - lo < eps:                                             # 防止分母过小导致数值不稳定
        return np.clip((x - lo), 0, 1)                            # 直接裁剪到 [0,1]
    x = (x - lo) / (hi - lo)                                      # 线性归一化到 [0,1]
    return np.clip(x, 0, 1)                                       # 进一步确保范围在 [0,1]

def srgb_gamma_encode(x):                                         # 线性空间 -> sRGB 伽马编码
    """线性 -> sRGB 伽马编码，x ∈ [0,1]"""                        # 函数文档字符串
    x = np.clip(x, 0, 1)                                          # 先裁到 [0,1]，防止出现负数或>1
    a = 0.055                                                     # sRGB 标准常数
    out = np.where(                                               # 分段函数：小于阈值走线性段，否则幂函数段
        x <= 0.0031308,
        12.92 * x,
        (1 + a) * np.power(x, 1/2.4) - a
    )
    return np.clip(out, 0, 1)                                     # 结果裁到 [0,1]

def put_mask(gray_or_rgb, mask_color=(255, 0, 0), beta=0.2):      # 在灰度或RGB图上叠加纯色蒙版
    """
    在灰度或RGB图上叠加纯色蒙版。mask_color 为 (R,G,B) 0-255，beta 混合比。
    返回 uint8 RGB。
    """
    if gray_or_rgb.ndim == 2:                                     # 若输入是灰度图 (H,W)
        base = np.stack([gray_or_rgb]*3, axis=-1).astype(np.float32)  # 扩展为3通道并转 float32
    elif gray_or_rgb.ndim == 3 and gray_or_rgb.shape[2] == 3:     # 若输入已经是 RGB (H,W,3)
        base = gray_or_rgb.astype(np.float32)                     # 转 float32 便于计算
    else:                                                         # 其他形状一律视为不支持
        raise ValueError(f"Unexpected image shape: {gray_or_rgb.shape}")
    color = np.array(mask_color, dtype=np.float32).reshape(1,1,3) # 目标颜色转为 (1,1,3) 方便广播
    out = (1.0 - beta) * base + beta * color                      # (1-β)*原图 + β*纯色，实现半透明叠加
    return np.clip(out, 0, 255).astype(np.uint8)                  # 裁剪并转回 uint8

def getRGB(dWave, maxPix=1.0, gamma=1.0):                         # 单波长到近似RGB的可视化映射（非严格色度学）
    """按常见近似把单波长映射到 RGB（0-255），仅作可视化，不等同于严格色度学。"""
    waveArea = [380, 440, 490, 510, 580, 645, 780]                # 各区段的分界波长
    minusWave = [0,   440, 440, 510, 510, 645, 780]               # 该近似模型用到的参考值
    deltWave  = [1,    60,  50,  20,  70,  65,  35]               # 各区段的跨度
    for p in range(len(waveArea)):                                 # 找到 dWave 所在区段的索引 p
        if dWave < waveArea[p]:
            break
    pVar = abs(minusWave[p] - dWave) / deltWave[p]                 # 归一化的区段内位置
    rgbs = [                                                       # 不同区段的 RGB 组合（简化/经验式）
        [0, 0, 0],
        [pVar, 0, 1],
        [0, pVar, 1],
        [0, 1, pVar],
        [pVar, 1, 0],
        [1, pVar, 0],
        [1, 0, 0],
        [0, 0, 0],
    ]
    # 边缘衰减（人眼对可见光两端敏感度较低）
    if (dWave >= 380) and (dWave < 420):
        alpha = 0.3 + 0.7 * (dWave - 380) / (420 - 380)           # 线性上升
    elif (dWave >= 420) and (dWave < 701):
        alpha = 1.0                                                # 中间波段不衰减
    elif (dWave >= 701) and (dWave < 780):
        alpha = 0.3 + 0.7 * (780 - dWave) / (780 - 700)           # 线性下降
    else:
        return [255, 255, 255]                                     # 非可见区返回白色（可视化占位）
    return [int(maxPix * (c * alpha) ** gamma * 255) for c in rgbs[p]]  # 应用亮度与γ，输出0-255整数

# ===== 2) HSI -> RGB（用你给的 CIE1931 系数；做插值+鲁棒归一） =====
def HSI2RGB_function(bands_nm, hsi_cube):                          # 将 (B,H,W) 光谱立方体映射为 (H,W,3) RGB
    """
    bands_nm: (B,) 波长，单位 nm
    hsi_cube: (B, H, W) 光谱立方体，线性强度或反射率
    返回: (H, W, 3) 的 RGB，可直接 imshow
    """
    # ===== CIE1931 查表（块注释）：每一行是 [波长(nm), R权重, G权重, B权重] 的经验/拟合系数 =====
    CIE1931 = np.array([
        [380, 0.0272, -0.0115, 0.9843],
        [385, 0.0268, -0.0114, 0.9846],
        [390, 0.0263, -0.0114, 0.9851],
        [395, 0.0256, -0.0113, 0.9857],
        [400, 0.0247, -0.0112, 0.9865],
        [405, 0.0237, -0.0111, 0.9874],
        [410, 0.0225, -0.0109, 0.9884],
        [415, 0.0207, -0.0104, 0.9897],
        [420, 0.0181, -0.0094, 0.9913],
        [425, 0.0142, -0.0076, 0.9934],
        [430, 0.0088, -0.0048, 0.9960],
        [435, 0.0012, -0.0007, 0.9995],
        [440, -0.0084, 0.0018, 1.0036],
        [445, -0.0213, 0.0120, 1.0093],
        [450, -0.0390, 0.0218, 1.0172],
        [455, -0.0618, 0.0345, 1.0273],
        [460, -0.0909, 0.0517, 1.0392],
        [465, -0.1281, 0.0762, 1.0519],
        [470, -0.1821, 0.1175, 1.0646],
        [475, -0.2584, 0.1840, 1.0744],
        [480, -0.3667, 0.2906, 1.0761],
        [485, -0.5200, 0.4568, 1.0632],
        [490, -0.7150, 0.6996, 1.0154],
        [495, -0.9459, 1.0247, 0.9212],
        [500, -1.1685, 1.3905, 0.7780],
        [505, -1.3182, 1.7195, 0.5987],
        [510, -1.3371, 1.9318, 0.4053],
        [515, -1.2076, 1.9699, 0.2377],
        [520, -0.9830, 1.8534, 0.1296],
        [525, -0.7386, 1.6662, 0.0724],
        [530, -0.5159, 1.4761, 0.0398],
        [535, -0.3304, 1.3105, 0.0199],
        [540, -0.1707, 1.1628, 0.0079],
        [545, -0.0293, 1.0282, 0.0011],
        [550, 0.0974, 0.9051, -0.0025],
        [555, 0.2121, 0.7919, -0.0040],
        [560, 0.3164, 0.6881, -0.0045],
        [565, 0.4112, 0.5932, -0.0044],
        [570, 0.4973, 0.5067, -0.0040],
        [575, 0.5751, 0.4283, -0.0034],
        [580, 0.6449, 0.3579, -0.0028],
        [585, 0.7071, 0.2952, -0.0023],
        [590, 0.7617, 0.2402, -0.0019],
        [595, 0.8087, 0.1928, -0.0015],
        [600, 0.8475, 0.1537, -0.0012],
        [605, 0.8800, 0.1209, -0.0009],
        [610, 0.9059, 0.0949, -0.0008],
        [615, 0.9265, 0.0741, -0.0006],
        [620, 0.9425, 0.0580, -0.0005],
        [625, 0.9550, 0.0454, -0.0004],
        [630, 0.9649, 0.0354, -0.0003],
        [635, 0.9730, 0.0272, -0.0002],
        [640, 0.9797, 0.0205, -0.0002],
        [645, 0.9850, 0.0152, -0.0002],
        [650, 0.9888, 0.0113, -0.0001],
        [655, 0.9918, 0.0083, -0.0001],
        [660, 0.9940, 0.0061, -0.0001],
        [665, 0.9954, 0.0047, -0.0001],
        [670, 0.9966, 0.0035, -0.0001],
        [675, 0.9975, 0.0025, 0.0000],
        [680, 0.9984, 0.0016, 0.0000],
        [685, 0.9991, 0.0009, 0.0000],
        [690, 0.9996, 0.0004, 0.0000],
        [695, 0.9999, 0.0001, 0.0000],
        [700, 1.0000, 0.0000, 0.0000],
        [705, 1.0000, 0.0000, 0.0000],
        [710, 1.0000, 0.0000, 0.0000],
        [715, 1.0000, 0.0000, 0.0000],
        [720, 1.0000, 0.0000, 0.0000],
        [725, 1.0000, 0.0000, 0.0000],
        [730, 1.0000, 0.0000, 0.0000],
        [735, 1.0000, 0.0000, 0.0000],
        [740, 1.0000, 0.0000, 0.0000],
        [745, 1.0000, 0.0000, 0.0000],
        [750, 1.0000, 0.0000, 0.0000],
        [755, 1.0000, 0.0000, 0.0000],
        [760, 1.0000, 0.0000, 0.0000],
        [765, 1.0000, 0.0000, 0.0000],
        [770, 1.0000, 0.0000, 0.0000],
        [775, 1.0000, 0.0000, 0.0000],
        [780, 1.0000, 0.0000, 0.0000],
    ])
    wl = CIE1931[:, 0]                                            # 提取波长列 (N,)
    print("wl_shape", wl.shape)                                   # 打印波长列形状，调试用
    M  = CIE1931[:, 1:]                                           # 提取 RGB 权重矩阵 (N,3)
    print("M.shape", M.shape)                                     # 打印权重矩阵形状，调试用

    bands_nm = np.asarray(bands_nm).astype(float)                 # 确保波长为 float 数组
    hsi = np.asarray(hsi_cube)                                    # 确保 HSI 立方体为 numpy 数组

    # 只用 380–780nm 范围内的波段，并用插值拿到每个 band 对应的 3×1 系数
    in_range = (bands_nm >= wl.min()) & (bands_nm <= wl.max())    # 标记哪些输入波长落在表的范围内
    print("in_range", in_range.shape)                              # 打印布尔掩码形状，调试用
    if not np.any(in_range):                                      # 若没有任何波段在范围内
        raise ValueError("No bands fall within [380, 780] nm.")   # 抛出异常提示
    bands_sel = bands_nm[in_range]                                # 选择有效的波段值 (B_sel,)
    hsi_sel   = hsi[in_range, :, :]                               # 同步选择对应的谱面 (B_sel,H,W)
    print("his_sel.shape", hsi_sel.shape)                         # 打印选择后的 HSI 形状

    # 对每一列做线性插值，得到 (B_sel,3) 的系数
    coeff = np.column_stack([                                     # 逐列插值并按列堆叠为 (B_sel,3)
        np.interp(bands_sel, wl, M[:, i]) for i in range(3)
    ])  # (B_sel,3)

    # 把 (B_sel,H,W) · (B_sel,3) -> (H,W,3)
    rgb_lin = np.tensordot(hsi_sel, coeff, axes=(0, 0))           # 张量点积：对波段维求和得到3通道
    rgb_lin = np.clip(rgb_lin, 0, None)                           # 负值裁零，避免出现负亮度

    # 分位数归一 + γ 编码（可选）
    rgb_lin = percentile_normalize(rgb_lin, 1, 99)                # 对 RGB 张量做分位数归一化
    rgb_out = srgb_gamma_encode(rgb_lin)                          # 应用 sRGB 伽马编码便于显示

    return rgb_out                                                # 返回 (H,W,3)，float，范围[0,1]

# ===== 3) 单波段着色叠加并保存 =====
def save_band_overlays(hsi_R, bands, save_image_path, data_name,  # 生成每个波段的着色叠加图并保存
                       select_bands=np.arange(400, 1001, 10),     # 选择需要可视化的波段列表（默认 400~1000 每10nm）
                       beta=0.2):                                  # 叠加透明度
    """
    hsi_R: (B,H,W) 反射率/强度，范围建议 [0,1]
    bands: (B,) 波长列表（nm）
    """
    os.makedirs(save_image_path, exist_ok=True)                   # 确保输出目录存在
    H, W = hsi_R.shape[-2], hsi_R.shape[-1]                       # 取得图像高宽（未直接使用，仅示意）

    for band_nm in select_bands:                                  # 遍历每个要可视化的波长
        idx = int(np.argmin(np.abs(bands - band_nm)))             # 找到最接近该波长的通道索引
        ch  = hsi_R[idx, :, :]                                    # 取出对应通道的二维图 (H,W)
        ch01 = percentile_normalize(ch, 1, 99)                    # 分位数归一化到 [0,1]
        pltim = np.uint8(np.clip(ch01 * 255.0, 0, 255))           # 转为 0-255 的 uint8 灰度图

        color = tuple(getRGB(float(band_nm)))                     # 获取该波长的可视化颜色 (R,G,B)
        # print("color_shape", )
        im = put_mask(pltim, mask_color=color, beta=beta)         # 叠加彩色蒙版得到 RGB 图

        fig = plt.figure()                                        # 新建画布
        plt.imshow(im)                                            # 显示 RGB 图像（不使用 cmap）
        plt.axis('off')                                           # 关闭坐标轴
        fig.set_size_inches(700/300.0, 700/300.0)                 # 设置物理尺寸：700px @ 300dpi
        plt.subplots_adjust(0,0,1,1)                              # 去除四周空白

        out_path = os.path.join(                                  # 拼出输出文件路径
            save_image_path, f"{data_name}_{int(band_nm)}_hsi_R.png"
        )
        fig.savefig(out_path, transparent=True, dpi=300,bbox_inches='tight', pad_inches=0)  # 以透明背景保存
        plt.close(fig)                                            # 关闭当前图，释放内存

# ===== 4) 示例用法（按你的变量名习惯；实际跑时取消注释并提供数据） =====
if __name__ == "__main__":
    # path = r"E:\quwu\Unet_Dehazing-main\dataset\PolHSI\building\clear\HSI_R_Image_20250812151024272_pol_all.h5"  # .h5 文件路径
    # with h5py.File(path, "r") as f:                                   # 以只读方式打开 h5 文件
    #     print("可用数据集(keys):", list(f.keys()))                     # 打印所有数据集名称，便于确认真实键名
    #     key_candidates = ["hsi_R", "hsi_Hazy", "HSI_R", "HSR_R"]      # 可能的键名候选列表（根据你的数据命名习惯）
    #     key = next((k for k in key_candidates if k in f.keys()), None) # 找到第一个存在的键名
    #     if key is None:                                               # 如果都没找到
    #         raise KeyError("文件内未找到 hsi_R/hsr_R 等数据集，请打印 keys 确认真实名称。")  # 抛错提示
    #     hsi_R = f[key][...]                                           # 读成 numpy 数组（例如形状 (4,61,H,W)）
    npy_path = r"/data/ppm/dehazing_PPM/Test_Results/lunwen_0001/npy/sample_000000.npy"
    # 或者 .npz: npy_path = r".../sample_000000.npz"

    # 读取文件（兼容 .npz 和 .npy（字典或数组））
    data_loaded = np.load(npy_path, allow_pickle=True).item()  # 读取 .npy 文件内容
    print("data_loaded type:", type(data_loaded))  # 打印加载对象类型，调试用
    # 目标变量名：尽量匹配你后续要用的 hsi_R 名称
    hsi_R = None



    hsi_R = data_loaded["clear"]
    print("hsi_R.shape", hsi_R.shape, type(hsi_R)) 
    
    b_idx = 0                                                         # 选择第 b_idx 个样本（例如 B=4 里的第0个）
    hsi_R = hsi_R[b_idx]                                              # 取出该样本 -> 形状 (C=61, H, W)
    bands = np.arange(400, 1001, 10).astype(float)                    # 构造波长数组 400~1000nm，每10nm一段
    print("bands.shape", bands.shape, type(bands))                    # 打印波长数组信息
    print("hsi_R.shape", hsi_R.shape, type(hsi_R))                    # 打印该样本的 HSI 形状信息
    save_image_path = "./band_vis_clear/sample0"                                   # 输出目录
    data_name = "sample0"                                            # 输出文件名前缀
    # exit("ppm")

    # A) 生成一张 HSI 合成 RGB 可视化
    rgb = HSI2RGB_function(bands, hsi_R)                              # 调用 HSI->RGB 函数得到 (H,W,3)
    plt.figure(figsize=(6,6))                                         # 新建画布（6x6英寸）
    plt.imshow(rgb)                                                   # 显示 RGB
    plt.axis('off')                                                   # 关闭坐标轴
    plt.show()    
    out_path = "/data/ppm/dehazing_PPM/band_vis_clear/sample0/sample_000000_rgb.png"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close()
    print("Saved image to", out_path)                                                    # 显示窗口（脚本环境下可能会阻塞）

    # B) 存各个波段着色叠加图（这里示例 400–1000nm，每10nm一张）
    save_band_overlays(                                               # 批量生成并保存每个波段的伪彩叠加图
        hsi_R, bands, save_image_path, data_name,
        select_bands=np.arange(400, 1001, 10),                        # 选择的波段范围
        beta=0.2                                                      # 叠加透明度
    )
