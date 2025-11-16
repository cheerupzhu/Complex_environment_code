import os          # 导入操作文件和目录的模块
import h5py        # 导入用于读取 .h5 文件的库

# 要遍历的目录：Output 文件夹
input_dir = r"/data/ppm/dehazing_PPM/PPM_hazy_data1/data1/clear"

# 遍历文件夹下的所有文件
for file_name in os.listdir(input_dir):
    print("Processing file: ", file_name)
    # 判断是否是 .h5 文件
    if file_name.endswith(".h5"):
        # 拼接完整的文件路径
        file_path = os.path.join(input_dir, file_name)

        # 打印当前处理的文件名
        print(f"\n=== {file_path} ===")

        # 打开 .h5 文件
        with h5py.File(file_path, "r") as f:
            

            # 定义一个函数，用于输出文件中的结构信息
            def print_info(name, obj):
                # 如果是数据集，输出名称、形状和数据类型
                if isinstance(obj, h5py.Dataset):
                    print(f"{name} | shape={obj.shape} | dtype={obj.dtype}")
                # 如果是组，只输出名称
                elif isinstance(obj, h5py.Group):
                    print(f"{name} (Group)")

            # 遍历文件中的所有对象并打印信息
            f.visititems(print_info)

# 全部处理完成后，打印提示
print("✅ 已经遍历完 Output 文件夹下所有的 .h5 文件。")
