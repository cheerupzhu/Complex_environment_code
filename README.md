# Complex_environment_code
This contains complex environment code.(I hope it can inspire you.)


🚀数据类型（.h5）有两个键hsi_R和mos  

hsi_R | shape=(4, 61, 1024, 1224) | dtype=float32

mos | shape=(4, 1024, 1224) | dtype=float32

通过网盘分享的文件：dataset
链接: https://pan.baidu.com/s/18ft8LqYIkeyo3NdOdP6U0w?pwd=hyrn 提取码: hyrn 
--来自百度网盘超级会员v1的分享


---------------------------------------------------------------------------

在model里面利用以下代码 即可很好查看网络结构

	
    with open('model_structure.txt', 'w', encoding='utf-8') as f:
        for name, layer in model.named_children():
            f.write(f"{name}: {layer}\n")
            f.write("*" * 50 + "\n")  # 分隔线


在model里面利用以下代码 即可很好查看网络参数

    with open('model_params.txt', 'w', encoding='utf-8') as f:
        for name, layer in model.state_dict().items():
            f.write(f"{name}: {layer.shape}\n")
            f.write("*" * 50 + "\n")  # 分隔线


更加清晰
	from torchinfo import summary
	summary(model,input_size=(batch_size，244，100，100))
