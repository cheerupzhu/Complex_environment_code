import os
import h5py
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class H5HazyClearDataset(Dataset):
    def __init__(self, root_dir, patch_size ,arg=False):
        """
        root_dir: 数据集根目录，比如 "E:/quwu/Unet_Dehazing-main/dataset/PolHSI"
        patch_size: (ph, pw)，要裁剪的 patch 大小
        arg: 可选的数据增强函数
        """
        self.samples = []  # 保存所有 (hazy_path, clear_path)
        self.arg = arg
        self.patch_size = patch_size

        # 遍历 building/car/plant/sky
        for category in os.listdir(root_dir):
            cat_path = os.path.join(root_dir, category)
            hazy_dir = os.path.join(cat_path, "hazy")
            clear_dir = os.path.join(cat_path, "clear")

            if not os.path.isdir(hazy_dir) or not os.path.isdir(clear_dir):
                continue  # 跳过不符合结构的文件夹

            hazy_files = sorted([os.path.join(hazy_dir, f) for f in os.listdir(hazy_dir) if f.endswith(".h5")])
            clear_files = sorted([os.path.join(clear_dir, f) for f in os.listdir(clear_dir) if f.endswith(".h5")])

            assert len(hazy_files) == len(clear_files), f"{category} 下 hazy 和 clear 数量不一致！"

            # 保存成对路径
            self.samples.extend(list(zip(hazy_files, clear_files)))

        print(f"共加载 {len(self.samples)} 对样本。")


    def __len__(self):
        return len(self.samples)

    def random_crop_pair(self, arr1, arr2, ph, pw):
        """在 (C,H,W) 上用同一个随机窗口裁剪两个数组"""
        _, H, W = arr1.shape
        if H == ph and W == pw:
            return arr1, arr2
        top = np.random.randint(0, H - ph + 1)
        left = np.random.randint(0, W - pw + 1)
        return (arr1[:, top:top + ph, left:left + pw],
                arr2[:, top:top + ph, left:left + pw])

    def augment(self, inputs, targets):
        """
        数据增强：随机水平翻转、垂直翻转、90/180/270°旋转
        确保 hazy(inputs) 和 clear(targets) 同步操作
        inputs, targets: torch.Tensor, shape=(C,H,W)
        """
        # 随机水平翻转
        if random.random() < 0.5:
            inputs = torch.flip(inputs, dims=[2])  # 宽度方向
            targets = torch.flip(targets, dims=[2])

        # 随机垂直翻转
        if random.random() < 0.5:
            inputs = torch.flip(inputs, dims=[1])  # 高度方向
            targets = torch.flip(targets, dims=[1])

        # 随机旋转 90/180/270 度
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])  # k=1:90°, k=2:180°, k=3:270°
            inputs = torch.rot90(inputs, k, dims=[1, 2])
            targets = torch.rot90(targets, k, dims=[1, 2])

        return inputs, targets

    def __getitem__(self, idx):
        hazy_path, clear_path = self.samples[idx]

        # 读取 H5 文件
        with h5py.File(hazy_path, "r") as f:
            hazy = f[list(f.keys())[0]][:]  # 取第一个 dataset
        with h5py.File(clear_path, "r") as f:
            clear = f[list(f.keys())[0]][:]
        # print("hazy" , hazy.shape, hazy.dtype, hazy.max(), hazy.min(), hazy.mean())
        # print("clear" , clear.shape, clear.dtype, clear.max(),clear.min(),clear.mean())
        # exit("ppm")
        hazy_0 = hazy[0] #偏振0°下的61通道 （61，H，W）
        hazy_1 = hazy[1]
        hazy_2 = hazy[2]
        hazy_3 = hazy[3]
        inputs = np.concatenate([hazy_0, hazy_1, hazy_2, hazy_3], axis=0)  # shape = (244, H, W)

        clear_0 = clear[0] #偏振0°下的61通道 （61，H，W）
        clear_1 = clear[1]
        clear_2 = clear[2]
        clear_3 = clear[3]
        targets = np.concatenate([clear_0, clear_1, clear_2, clear_3], axis=0)  # (244,H,W)

        # 随机裁剪 patch
        ph, pw = self.patch_size
        inputs, targets = self.random_crop_pair(inputs, targets, ph, pw)

        # 转换成 torch tensor
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        # print("inputs:", inputs.shape, inputs.dtype,
        #       inputs.max().item(), inputs.min().item(), inputs.mean().item())
        # print("targets:", targets.shape, targets.dtype,
        #       targets.max().item(), targets.min().item(), targets.mean().item())

        if self.arg:  #数据增强
            inputs, targets = self.augment(inputs, targets)

        return {"hazy": inputs, "clear": targets}

    # -*- coding: utf-8 -*-  # 指定编码，确保中文注释正常显示
    import os  # 操作系统路径与目录操作
    import numpy as np  # 数组读写与拼接
    import h5py  # 读取 .h5 文件
    import torch  # 张量与深度学习框架
    from torch.utils.data import Dataset  # PyTorch 数据集基类

    class H5HazyClearTestDataset(Dataset):  # 定义专用于“测试阶段”的数据集类（不裁剪、不增强）
        def __init__(self, root_dir):  # 初始化，传入测试数据集的根目录
            """
            root_dir: 数据集根目录，比如 "E:/quwu/Unet_Dehazing-main/dataset/PolHSI"
            该测试集类只做“成对文件的读取与拼接”，不做任何裁剪或数据增强
            """
            self.samples = []  # 用于保存所有 (hazy_path, clear_path) 的成对列表

            # 遍历类别子文件夹（如 building/car/plant/sky），与训练集保持同样目录结构
            for category in os.listdir(root_dir):  # 枚举根目录下的所有子目录
                cat_path = os.path.join(root_dir, category)  # 拼接得到子目录完整路径
                hazy_dir = os.path.join(cat_path, "hazy")  # hazy 数据所在目录
                clear_dir = os.path.join(cat_path, "clear")  # clear 数据所在目录

                if not os.path.isdir(hazy_dir) or not os.path.isdir(clear_dir):  # 若子目录结构不完整
                    continue  # 跳过该子目录

                # 收集所有以 .h5 结尾的文件，并保持排序一致，确保 hazy/clear 成对对应
                hazy_files = sorted(
                    [os.path.join(hazy_dir, f) for f in os.listdir(hazy_dir) if f.endswith(".h5")])  # hazy 文件列表
                clear_files = sorted(
                    [os.path.join(clear_dir, f) for f in os.listdir(clear_dir) if f.endswith(".h5")])  # clear 文件列表

                assert len(hazy_files) == len(clear_files), f"{category} 下 hazy 和 clear 数量不一致！"  # 断言两者数量匹配

                # 将 (hazy_path, clear_path) 以元组形式加入 samples 列表
                self.samples.extend(list(zip(hazy_files, clear_files)))  # 扩展全局样本列表

            print(f"测试集共加载 {len(self.samples)} 对样本。")  # 打印总样本对数量，便于确认

        def __len__(self):  # 返回数据集大小
            return len(self.samples)  # 样本对数量

        def __getitem__(self, idx):  # 根据索引返回一对样本
            hazy_path, clear_path = self.samples[idx]  # 取出索引对应的 hazy 与 clear 的路径

            # 读取 hazy 的 .h5 文件，假设每个 .h5 只有一个数据集条目，取其第一个键
            with h5py.File(hazy_path, "r") as f:  # 打开 hazy h5 文件
                hazy = f[list(f.keys())[0]][:]  # 读取对应数据为 numpy 数组，形状期望为 (4, 61, H, W)

            # 读取 clear 的 .h5 文件，方式同上
            with h5py.File(clear_path, "r") as f:  # 打开 clear h5 文件
                clear = f[list(f.keys())[0]][:]  # 读取对应数据为 numpy 数组，形状期望为 (4, 61, H, W)

            # 拆分四个偏振分量，并在“通道维”上拼接（得到 244 通道：4*61）
            hazy_0 = hazy[0]  # 偏振 0°，形状 (61, H, W)
            hazy_1 = hazy[1]  # 偏振 45°，形状 (61, H, W) —— 若数据标注不同，请按你的实际注释
            hazy_2 = hazy[2]  # 偏振 90°，形状 (61, H, W)
            hazy_3 = hazy[3]  # 偏振 135°，形状 (61, H, W)
            inputs = np.concatenate([hazy_0, hazy_1, hazy_2, hazy_3], axis=0)  # 在通道维拼接，得到 (244, H, W)

            clear_0 = clear[0]  # 偏振 0°，形状 (61, H, W)
            clear_1 = clear[1]  # 偏振 45°，形状 (61, H, W)
            clear_2 = clear[2]  # 偏振 90°，形状 (61, H, W)
            clear_3 = clear[3]  # 偏振 135°，形状 (61, H, W)
            targets = np.concatenate([clear_0, clear_1, clear_2, clear_3], axis=0)  # 在通道维拼接，得到 (244, H, W)

            # 将 numpy 数组转换为 torch.float32 张量
            inputs = torch.tensor(inputs, dtype=torch.float32)  # 转为 float32 的输入张量
            targets = torch.tensor(targets, dtype=torch.float32)  # 转为 float32 的标签张量

            # 这里不做任何随机裁剪或数据增强，保持整幅图像尺寸输入模型
            # 若模型需要归一化到 [0,1]，请确保 .h5 中数据已在该范围；否则可在此处做缩放（例如除以 65535.）

            return {"hazy": inputs, "clear": targets}  # 返回字典，键名与训练/测试脚本保持一致


class H5HazyClearTestDataset(Dataset):  # 定义专用于“测试阶段”的数据集类（不裁剪、不增强）
    def __init__(self, root_dir):  # 初始化，传入测试数据集的根目录
        """
        root_dir: 数据集根目录，比如 "E:/quwu/Unet_Dehazing-main/dataset/PolHSI"
        该测试集类只做“成对文件的读取与拼接”，不做任何裁剪或数据增强
        """
        self.samples = []  # 用于保存所有 (hazy_path, clear_path) 的成对列表

        # 遍历类别子文件夹（如 building/car/plant/sky），与训练集保持同样目录结构
        for category in os.listdir(root_dir):  # 枚举根目录下的所有子目录
            cat_path = os.path.join(root_dir, category)  # 拼接得到子目录完整路径
            hazy_dir = os.path.join(cat_path, "hazy")  # hazy 数据所在目录
            clear_dir = os.path.join(cat_path, "clear")  # clear 数据所在目录

            if not os.path.isdir(hazy_dir) or not os.path.isdir(clear_dir):  # 若子目录结构不完整
                continue  # 跳过该子目录

            # 收集所有以 .h5 结尾的文件，并保持排序一致，确保 hazy/clear 成对对应
            hazy_files = sorted([os.path.join(hazy_dir, f) for f in os.listdir(hazy_dir) if f.endswith(".h5")])  # hazy 文件列表
            clear_files = sorted([os.path.join(clear_dir, f) for f in os.listdir(clear_dir) if f.endswith(".h5")])  # clear 文件列表

            assert len(hazy_files) == len(clear_files), f"{category} 下 hazy 和 clear 数量不一致！"  # 断言两者数量匹配

            # 将 (hazy_path, clear_path) 以元组形式加入 samples 列表
            self.samples.extend(list(zip(hazy_files, clear_files)))  # 扩展全局样本列表

        print(f"测试集共加载 {len(self.samples)} 对样本。")  # 打印总样本对数量，便于确认

    def __len__(self):  # 返回数据集大小
        return len(self.samples)  # 样本对数量

    def __getitem__(self, idx):  # 根据索引返回一对样本
        hazy_path, clear_path = self.samples[idx]  # 取出索引对应的 hazy 与 clear 的路径

        # 读取 hazy 的 .h5 文件，假设每个 .h5 只有一个数据集条目，取其第一个键
        with h5py.File(hazy_path, "r") as f:  # 打开 hazy h5 文件
            hazy = f[list(f.keys())[0]][:]  # 读取对应数据为 numpy 数组，形状期望为 (4, 61, H, W)

        # 读取 clear 的 .h5 文件，方式同上
        with h5py.File(clear_path, "r") as f:  # 打开 clear h5 文件
            clear = f[list(f.keys())[0]][:]  # 读取对应数据为 numpy 数组，形状期望为 (4, 61, H, W)

        # 拆分四个偏振分量，并在“通道维”上拼接（得到 244 通道：4*61）
        hazy_0 = hazy[0]  # 偏振 0°，形状 (61, H, W)
        hazy_1 = hazy[1]  # 偏振 45°，形状 (61, H, W) —— 若数据标注不同，请按你的实际注释
        hazy_2 = hazy[2]  # 偏振 90°，形状 (61, H, W)
        hazy_3 = hazy[3]  # 偏振 135°，形状 (61, H, W)
        inputs = np.concatenate([hazy_0, hazy_1, hazy_2, hazy_3], axis=0)  # 在通道维拼接，得到 (244, H, W)

        clear_0 = clear[0]  # 偏振 0°，形状 (61, H, W)
        clear_1 = clear[1]  # 偏振 45°，形状 (61, H, W)
        clear_2 = clear[2]  # 偏振 90°，形状 (61, H, W)
        clear_3 = clear[3]  # 偏振 135°，形状 (61, H, W)
        targets = np.concatenate([clear_0, clear_1, clear_2, clear_3], axis=0)  # 在通道维拼接，得到 (244, H, W)

        # 将 numpy 数组转换为 torch.float32 张量
        inputs = torch.tensor(inputs, dtype=torch.float32)  # 转为 float32 的输入张量
        targets = torch.tensor(targets, dtype=torch.float32)  # 转为 float32 的标签张量

        # 这里不做任何随机裁剪或数据增强，保持整幅图像尺寸输入模型
        # 若模型需要归一化到 [0,1]，请确保 .h5 中数据已在该范围；否则可在此处做缩放（例如除以 65535.）

        return {"hazy": inputs, "clear": targets}  # 返回字典，键名与训练/测试脚本保持一致

# class PolarizationDataset(Dataset):
#     def __init__(self, root_dir, patch_size,augment=False):
#         """
#         Args:
#             root_dir (string): Root directory containing 'inputs' and 'labels' subdirectories.
#             transform (callable, optional): Transformations to apply to input tensors.
#             augment (bool): Whether to apply data augmentation.
#         """
#         self.arg = augment
#         self.root_dir = root_dir
#         self.patch_size = patch_size
#
#         self.input_dir = os.path.join(root_dir, 'inputs')
#         self.label_dir = os.path.join(root_dir, 'labels')
#
#         # Collect all sample paths
#         self.samples = []
#         for object_name in sorted(os.listdir(self.input_dir)):
#             object_input_dir = os.path.join(self.input_dir, object_name)
#             object_label_dir = os.path.join(self.label_dir, object_name)
#             # object_mask_dir  = os.path.join(self.label_dir, object_name)
#
#             if not os.path.isdir(object_input_dir):
#                 continue  # Skip non-directory files
#
#             for sample_name in sorted(os.listdir(object_input_dir)):
#                 sample_input_dir = os.path.join(object_input_dir, sample_name)
#                 sample_label_dir = os.path.join(object_label_dir, sample_name)
#                 # sample_mask_dir = os.path.join(object_input_dir, object_name)
#                 # label_file = os.path.join(sample_label_dir, f"{sample_name}_height_and_normal_map.npy")
#
#                 if os.path.isdir(sample_input_dir) and os.path.isdir(sample_label_dir):
#                     self.samples.append({
#                         'object': object_name,
#                         'sample': sample_name,
#                         'input_dir': sample_input_dir,
#                         'label_dir': sample_label_dir,
#                     })
#
#     def arguement(self, img, rotTimes, vFlip, hFlip):
#         # Random rotation
#         for j in range(rotTimes):
#             img = np.rot90(img.copy(), axes=(1, 2))
#         # Random vertical Flip
#         for j in range(vFlip):
#             img = img[:, :, ::-1].copy()
#         # Random horizontal Flip
#         for j in range(hFlip):
#             img = img[:, ::-1, :].copy()
#         return img
#
#     # def arguement(input_img, label_img, rotTimes, vFlip, hFlip):
#     #     # 对输入图像进行翻转操作
#     #     for j in range(rotTimes):
#     #         input_img = np.rot90(input_img, axes=(1, 2))
#     #         label_img = np.rot90(label_img, axes=(0, 1))
#     #     for j in range(vFlip):
#     #         input_img = input_img[:, :, ::-1]
#     #         label_img = label_img[:, ::-1]
#     #     for j in range(hFlip):
#     #         input_img = input_img[:, ::-1, :]
#     #         label_img = label_img[::-1, :]
#     #
#     #     return input_img, label_img
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         # -- 1、prepare paths
#         sample_info = self.samples[idx]
#         input_dir = sample_info['input_dir']
#         label_file = sample_info['label_dir']
#         # mask_dir = sample_info['mask_dir']
#
#         # -- 2、Load 4 polarization images
#         polarization_images = []
#         polarization_labels = []
#         angles = ['l0', 'l45', 'l90', 'l135']
#         for angle in angles:
#             img_path = os.path.join(input_dir, f"{sample_info['sample']}_{angle}.png")
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Single channel
#
#             # print('img', img.dtype, img.shape, img.max(), img.mean(), img.min())
#             polarization_images.append(img)
#
#
#         input_pols = np.array(polarization_images, np.float32)
#         input_pols = input_pols / input_pols.max()
#
#
#
#
#         # -- 3、 Load mask
#         # mask_path = os.path.join(input_dir, f"{sample_info['sample']}_mask.png")
#         # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Single channel
#         # print('mask', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())
#         # mask = np.expand_dims(mask, axis=0)  # 添加一个通道维度 [b,h,w]
#         # mask = mask.astype(np.float32)
#         # mask = mask / mask.max()
#         # if mask.max() is np.nan:
#         # print('sample_info[sample]',sample_info['sample'])
#         # exit()
#         # print('mask', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())
#         # plt.imshow(input_pols[0, :, :].squeeze(), cmap='gray')
#         # plt.show()
#         # plt.imshow(mask.squeeze(), cmap='gray')
#         # plt.show()
#
#         # input_pols_mask = mask * input_pols
#         # print('input_pols_mask', input_pols_mask.dtype, input_pols_mask.shape, input_pols_mask.max(), input_pols_mask.mean(), input_pols_mask.min())
#
#         # plt.imshow(input_pols_mask[0, :, :].squeeze(), cmap='gray')
#         # plt.show()
#
#         # exit()
#
#         # # 扩展维度，使其变为 (1, H, W)
#         # mask = np.expand_dims(mask, axis=0)  # 添加一个通道维度
#         # mask_tensor = transforms.ToTensor()(mask)  #[1, H , W]
#
#
#
#
#         # # -- 4、 no Combine input channels (4 polarization = 4 channels)
#         # input_images = polarization_images
#         # input_tensors = [transforms.ToTensor()(img) for img in input_images]  # Each [1, H, W]
#         # input_tensor = torch.cat(input_tensors, dim=0)      # [4, H, W]
#
#
#         # -- 4、Load labels
#         for angle in angles:
#             img_path = os.path.join(input_dir, f"{sample_info['sample']}_{angle}.png")
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Single channel
#
#             # print('img', img.dtype, img.shape, img.max(), img.mean(), img.min())
#             polarization_labels.append(img)
#
#         label_np = np.array(polarization_labels, np.float32)  # [ H, W, 4]
#         label_np = label_np / input_pols.max()
#         # print('label_np', label_np.dtype, label_np.shape, label_np.max(), label_np.mean(), label_np.min())
#
#
#         # plt.imshow(label_norm)
#         # plt.show()
#
#         # label_norm = label_norm.transpose(2, 0, 1)
#         # label_norm = label_norm / label_norm.max()
#         # print('label_norm', type(label_norm), label_norm.dtype, label_norm.shape, label_norm.max(), label_norm.mean(), label_norm.min())
#         # exit('ppm')
#
#
#         # label_hight = label_np[:, :, 0]
#
#         # plt.imshow(label_hight, cmap='gray')
#         # plt.show()
#
#         # label_hight = np.expand_dims(label_hight, 0)
#         # print('label_hight', label_hight.dtype, label_hight.shape, label_hight.max(), label_hight.mean(), label_hight.min())
#         # label_hight = label_hight - label_hight.min()
#
#         # label_hight = label_hight / label_hight.max()
#         # print('label_hight', label_hight.dtype, label_hight.shape, label_hight.max(), label_hight.mean(), label_hight.min())
#
#
#
#         if self.arg:
#             rotTimes = random.randint(0, 3)
#             vFlip = random.randint(0, 1)
#             hFlip = random.randint(0, 1)
#             input_pols = self.arguement(input_pols, rotTimes, vFlip, hFlip)
#             label_np   = self.arguement(input_pols, rotTimes, vFlip, hFlip)
#
#         # Data augmentation (synchronized transforms)
#         # if self.augment:
#         #     input_tensor, label_tensor, mask_tensor = self.apply_transforms(input_tensor, label_tensor, mask_tensor)
#
#         # Apply input transformations (e.g., normalization)
#         # if self.transform:
#         #     input_tensor = self.transform(input_tensor)
#         #     # Typically, do not transform labels for regression tasks
#
#         # exit()
#
#         return input_pols, label_np
#
# class PolarizationDataset_Test(Dataset):
#     def __init__(self, root_dir, transform=None, augment=False, mask_dir=None):
#         """
#         Args:
#             root_dir (string): Root directory containing 'inputs' and 'labels' subdirectories.
#             transform (callable, optional): Transformations to apply to input tensors.
#             augment (bool): Whether to apply data augmentation.
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#         self.mask_dir = mask_dir
#         self.augment = augment
#         self.input_dir = os.path.join(root_dir, 'inputs')
#         self.label_dir = os.path.join(root_dir, 'labels')
#
#         # Collect all sample paths
#         self.samples = []
#         for object_name in sorted(os.listdir(self.input_dir)):
#             object_input_dir = os.path.join(self.input_dir, object_name)
#             object_label_dir = os.path.join(self.label_dir, object_name)
#             # object_mask_dir  = os.path.join(self.label_dir, object_name)
#
#             if not os.path.isdir(object_input_dir):
#                 continue  # Skip non-directory files
#
#             for sample_name in sorted(os.listdir(object_input_dir)):
#                 sample_input_dir = os.path.join(object_input_dir, sample_name)
#                 sample_label_dir = os.path.join(object_label_dir, sample_name)
#                 sample_mask_dir = os.path.join(object_input_dir, object_name)
#                 label_file = os.path.join(sample_label_dir, f"{sample_name}_height_and_normal_map.npy")
#
#                 if os.path.isdir(sample_input_dir) and os.path.isfile(label_file):
#                     self.samples.append({
#                         'object': object_name,
#                         'sample': sample_name,
#                         'input_dir': sample_input_dir,
#                         'label_file': label_file,
#                         'mask_dir': mask_dir,
#                     })
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         # -- 1、prepare paths
#         sample_info = self.samples[idx]
#         input_dir = sample_info['input_dir']
#         label_file = sample_info['label_file']
#         mask_dir = sample_info['mask_dir']
#         name = sample_info['sample']
#
#         # -- 2、Load 4 polarization images
#         polarization_images = []
#         angles = ['l0', 'l45', 'l90', 'l135']
#         for angle in angles:
#             img_path = os.path.join(input_dir, f"{sample_info['sample']}_{angle}.png")
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Single channel
#
#             # print('img', img.dtype, img.shape, img.max(), img.mean(), img.min())
#             polarization_images.append(img)
#
#
#         input_pols = np.array(polarization_images, np.float32)
#         input_pols = input_pols / input_pols.max()
#
#
#
#         # -- 3、 Load mask
#         mask_path = os.path.join(input_dir, f"{sample_info['sample']}_mask.png")
#         # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Single channel
#         # print('mask', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())
#         mask = np.expand_dims(mask, axis=0)  # 添加一个通道维度
#         mask = mask.astype(np.float32)
#         mask = mask / mask.max()
#         # print('mask', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())
#         # plt.imshow(input_pols[0, :, :].squeeze(), cmap='gray')
#         # plt.show()
#         # plt.imshow(mask.squeeze(), cmap='gray')
#         # plt.show()
#
#         input_pols_mask = mask * input_pols
#         # print('input_pols_mask', input_pols_mask.dtype, input_pols_mask.shape, input_pols_mask.max(), input_pols_mask.mean(), input_pols_mask.min())
#
#         # -- 4、Load labels
#         label_np = np.load(label_file)  # [ H, W, 4]
#         label_np = label_np.astype(np.float32)
#         # print('label_np', label_np.dtype, label_np.shape, label_np.max(), label_np.mean(), label_np.min())
#
#         label_norm = label_np[:, :, 1:]
#
#         # plt.imshow(label_norm)
#         # plt.show()
#
#         label_norm = label_norm.transpose(2, 0, 1)
#         # print('label_norm', label_norm.dtype, label_norm.shape, label_norm.max(), label_norm.mean(), label_norm.min())
#
#
#         label_hight = label_np[:, :, 0]
#         label_hight = np.expand_dims(label_hight, 0)
#         # print('label_hight', label_hight.dtype, label_hight.shape, label_hight.max(), label_hight.mean(), label_hight.min())
#         label_hight = label_hight - label_hight.min()
#
#
#
#         return input_pols_mask, label_norm, label_hight, mask, name
