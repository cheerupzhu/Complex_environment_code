import cv2
import os
import numpy as np

# 输入图像路径
input_image_path = r"E:\quwu\Unet_Dehazing-main\NP_haze_daima\results\Image_20251011152854074.bmp"

# 输出文件夹路径
output_folder = r"E:\quwu\Unet_Dehazing-main\NP_haze_daima\results\polarized_angles"

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 读取图像（假设是单通道 RAW 数据）
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("无法读取图像，请检查路径是否正确")
    exit()

# 获取图像尺寸
height, width = image.shape

# 检查是否能被2整除（DOFP通常是2×2马赛克）
if height % 2 != 0 or width % 2 != 0:
    print("图像尺寸需要能被2整除")
    exit()

# 初始化四个角度的图像
angle_0 = np.zeros((height // 2, width // 2), dtype=np.uint8)
angle_45 = np.zeros((height // 2, width // 2), dtype=np.uint8)
angle_90 = np.zeros((height // 2, width // 2), dtype=np.uint8)
angle_135 = np.zeros((height // 2, width // 2), dtype=np.uint8)

# 遍历图像，提取四个偏振角度（假设标准DOFP排列）
for i in range(0, height, 2):
    for j in range(0, width, 2):
        # 提取2×2块中的四个角度
        angle_0[i//2, j//2] = image[i, j]        # 0°
        angle_45[i//2, j//2] = image[i, j+1]     # 45°
        angle_90[i//2, j//2] = image[i+1, j]     # 90°
        angle_135[i//2, j//2] = image[i+1, j+1]  # 135°

# 保存四个角度的图像
cv2.imwrite(os.path.join(output_folder, "angle_0.bmp"), angle_0)
cv2.imwrite(os.path.join(output_folder, "angle_45.bmp"), angle_45)
cv2.imwrite(os.path.join(output_folder, "angle_90.bmp"), angle_90)
cv2.imwrite(os.path.join(output_folder, "angle_135.bmp"), angle_135)

print(f"DOFP偏振图像已成功分割并保存到 {output_folder} 文件夹")