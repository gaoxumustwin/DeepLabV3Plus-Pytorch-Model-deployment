'''
DeepLabV3Plus-Pytorch ncnn python推理
----------------------------------------------
@作者: gx
@邮箱: gaoxukkk@qq.com
@创建日期：2024年11月15日
'''



import ncnn
import cv2
import numpy as np
import os
from PIL import Image

# 颜色映射生成函数
def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    
    for i in range(N):   
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


# 后处理函数
def post_process(logits, cmap):

    # 检查 logits 的维度
    if len(logits.shape) == 3:  # 如果是 (H, W)，说明只有单个类别输出
        pred = logits.argmax(axis=0)  # 直接选择最大类别
    elif len(logits.shape) == 4:  # 如果是 (1, C, H, W)，选择最大类别的索引
        pred = logits.argmax(axis=1)[0]  # 去掉 batch 维度
    else:
        raise ValueError("Unexpected logits shape")

    # 将类别索引转换为颜色 
    pred = pred.astype(np.uint8)  # 转为 NumPy 数组
    colorized_preds = cmap[pred]  # 应用颜色映射

    # 保存前几个调试输出
    # print(np.unique(pred))  # 打印预测类别的唯一值，查看分布  [ 0 16]

    # 转为 PIL 图像并保存
    colorized_preds = Image.fromarray(colorized_preds)
    colorized_preds.save(os.path.join("samples/", 'res.png'))  # 保存最终的彩色图像


# 使用ncnn进行推理的代码
def infer_ncnn(image_path, param_file, bin_file, output_path="samples/res.png"):

    # 加载ncnn模型
    net = ncnn.Net()
    result = net.load_param(param_file)
    if result == 0:
        print("Successfully loaded param file.")
    else:
        print(f"Failed to load param file. Error code: {result}")
    
    result = net.load_model(bin_file)
    if result == 0:
        print("Successfully loaded bin file.")
    else:
        print(f"Failed to load bin file. Error code: {result}")

    # 读取并预处理图像
    image = cv2.imread(image_path)

    input_data = ncnn.Mat.from_pixels(image, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 513, 513)
    
    mean = [0.485*255, 0.456*255, 0.406*255]
    std = [1/(0.229*255), 1/(0.224*255), 1/(0.225*255)]
    input_data.substract_mean_normalize(mean, std)

    # 执行推理
    ex = net.create_extractor()
    ex.input('images', input_data)  # "input" 是ncnn模型的输入名称
    ret, logits = ex.extract('output')  # "output" 是ncnn模型的输出名称
    if ret != 0:
        print(f"Error in extracting output. Return code: {ret}")
        return

    # # 从ncnn.Mat转换为numpy数组
    logits = np.array(logits) #  (21, 513, 513)

    # 加载颜色映射
    cmap = voc_cmap()

    # 后处理并保存结果
    post_process(logits, cmap)

    print(f"Segmentation result saved to {output_path}")


# 在主程序中调用推理函数
if __name__ == "__main__":
    image_path = "samples/1_image.png"  # 输入图像路径
    param_file = "checkpoints/deeplabv3+.param"  # 转换后的ncnn模型的.param文件
    bin_file = "checkpoints/deeplabv3+.bin"  # 转换后的ncnn模型的.bin文件
    infer_ncnn(image_path, param_file, bin_file)
