# pengpengzhu
# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def Get_AOP_DOP(pols):
    """
    输入 pols: (4, H, W) -> polar images at 0,45,90,135
    返回: dop (H,W) in [0,1], aop (H,W) in radians mapped to [0, pi]
    """
    pols = pols / np.maximum(pols.max(), 1e-12)
    assert pols.shape[0] == 4, "输入数据必须是 (4, H, W)"
    pol_0 = pols[0, :, :]
    pol_45 = pols[1, :, :]
    pol_90 = pols[2, :, :]
    pol_135 = pols[3, :, :]

    S0 = pol_0 + pol_90
    S1 = pol_0 - pol_90
    S2 = pol_45 - pol_135

    dop = np.sqrt(S1 ** 2 + S2 ** 2) / np.maximum(S0, 1e-5)
    dop = np.clip(dop, 0, 1)

    aop = 0.5 * np.arctan2(S2, S1)   # [-pi/2, pi/2]
    aop = np.where(aop < 0, aop + np.pi, aop)  # [0, pi]

    return dop, aop

def save_aop_dop_from_h5(h5_path, save_dir, dataset_name=None, prefix="mos"):
    """
    载入 h5 文件（shape 应为 (4, C, H, W)），计算每个通道的 AOP & DOP 并保存 png。
    - h5_path: h5 文件路径
    - save_dir: 保存目录（会自动创建）
    - dataset_name: h5 内 dataset key；None 则使用第一个 key
    - prefix: 文件名前缀
    """
    os.makedirs(save_dir, exist_ok=True)
    with h5py.File(h5_path, 'r') as f:
        if dataset_name is None:
            keys = list(f.keys())
            if len(keys) == 0:
                raise RuntimeError("h5 文件内没有 dataset。")
            dataset_name = keys[0]
            print(f"未指定 dataset_name，使用第一个 key: '{dataset_name}'")
        data = f[dataset_name][()]

    if data.ndim != 4 or data.shape[0] != 4:
        raise ValueError(f"期望数据形状 (4, C, H, W)，但实际为 {data.shape}")

    _, C, H, W = data.shape
    print(f"读取数据: channels = {C}, H = {H}, W = {W}")

    out_aop_dir = os.path.join(save_dir, "AOP")
    out_dop_dir = os.path.join(save_dir, "DOP")
    os.makedirs(out_aop_dir, exist_ok=True)
    os.makedirs(out_dop_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(h5_path))[0]

    for ch in range(C):
        pols_ch = data[:, ch, :, :]  # (4, H, W)
        dop, aop = Get_AOP_DOP(pols_ch)

        # 在文件顶部或函数内定义你想要的像素大小和 dpi
        TARGET_W = 1224
        TARGET_H = 1024
        FIG_DPI = 300  # 可以改为其他值，但必须与 figsize 配合：pixels = inches * dpi
        # --- 保存 AOP (degrees 0-180) ---
        aop_deg = np.degrees(aop)  # 0..180

        fig = plt.figure(figsize=(TARGET_W / FIG_DPI, TARGET_H / FIG_DPI), dpi=FIG_DPI)
        # 让轴填满整个画布（无边距）：使用 add_axes 全覆盖
        ax = fig.add_axes([0, 0, 1, 1])
        # 注意：aop_deg 的单位是度，vmin/vmax 应与度一致
        im = ax.imshow(aop_deg, cmap='hsv', vmin=0, vmax=180)
        # 若你要显示 colorbar，请注意 colorbar 会改变图像区域并可能影响像素分布，
        # 因此这里为了保证输出像素精确且无边框，建议不显示 colorbar 或把 colorbar 单独保存。
        ax.axis('off')
        save_name_aop = f"{base_name}_{prefix}_AOP_ch{ch:02d}.png"
        save_path_aop = os.path.join(out_aop_dir, save_name_aop)
        plt.savefig(save_path_aop, dpi=FIG_DPI, transparent=True, pad_inches=0)
        plt.close(fig)

        # --- 保存 DOP (0-1) ---
        fig2 = plt.figure(figsize=(TARGET_W / FIG_DPI, TARGET_H / FIG_DPI), dpi=FIG_DPI)
        ax2 = fig2.add_axes([0, 0, 1, 1])
        im2 = ax2.imshow(dop, cmap='jet', vmin=0, vmax=1)
        ax2.axis('off')
        save_name_dop = f"{base_name}_{prefix}_DOP_ch{ch:02d}.png"
        save_path_dop = os.path.join(out_dop_dir, save_name_dop)
        plt.savefig(save_path_dop, dpi=FIG_DPI, transparent=True, pad_inches=0)
        plt.close(fig2)
        # # --- 保存 AOP (degrees 0-180) ---
        # aop_deg = np.degrees(aop)  # 0..180
        # fig = plt.figure(figsize=(6, 6), dpi=300)
        # ax = fig.add_subplot(111)
        # im = ax.imshow(aop_deg, cmap='hsv', vmin=0, vmax=np.pi)
        # cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cbar.set_label('Angle of Polarization (AOP) [degrees]')
        # ax.set_title(f'{prefix} AOP - channel {ch:02d}')
        # ax.axis('off')
        # save_name_aop = f"{base_name}_{prefix}_AOP_ch{ch:02d}.png"
        # save_path_aop = os.path.join(out_aop_dir, save_name_aop)
        # plt.savefig(save_path_aop, transparent=True, dpi=300, pad_inches=0)
        # plt.close(fig)
        #
        # # --- 保存 DOP (0-1) ---
        # fig2 = plt.figure(figsize=(6, 6), dpi=300)
        # ax2 = fig2.add_subplot(111)
        # im2 = ax2.imshow(dop, cmap='jet', vmin=0, vmax=1)
        # # cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        # # cbar2.set_label('Degree of Polarization (DOP)')
        # # ax2.set_title(f'{prefix} DOP - channel {ch:02d}')
        # ax2.axis('off')
        # save_name_dop = f"{base_name}_{prefix}_DOP_ch{ch:02d}.png"
        # save_path_dop = os.path.join(out_dop_dir, save_name_dop)
        # plt.savefig(save_path_dop, transparent=True, dpi=300, pad_inches=0)
        # plt.close(fig2)

        if (ch + 1) % 10 == 0 or (ch + 1) == C:
            print(f"已保存 {ch + 1}/{C} 个通道 (AOP & DOP) -> last saved: {save_path_dop}")

    print(f"全部保存完成，AOP 存放: {out_aop_dir}\nDOP 存放: {out_dop_dir}")

if __name__ == "__main__":
    # ========== 请按需修改下面路径 ==========
    h5_path = r"E:\quwu\Unet_Dehazing-main\NP_haze_daima\results\sample_000000_pred.h5"
    # h5_path = r"F:\dehazing\dataset\Test_true_hazy\output_images_New\HSI_R_Image_20250928171806317_pol_all.h5"
    # h5_path = r"F:\dehazing\dataset\Test_true_hazy\output_images_New\HSI_R_Image_20250928172157186_pol_all.h5"
    dataset_name = None  # 若知道内部 key，可填 'mos_all' 等
    save_image_path = r"E:\quwu\Unet_Dehazing-main\NP_haze_daima\results\DOP\results"
    prefix = "mos"
    # =========================================

    save_aop_dop_from_h5(h5_path, save_image_path, dataset_name=dataset_name, prefix=prefix)
