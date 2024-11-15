import onnxruntime
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


# 图像预处理函数（与PyTorch中相同）
def preprocess(image):
    # 转换为RGB
    image = image[..., ::-1]

    # 将像素值归一化到[0, 1]
    image = (image / 255.0).astype(np.float32)

    # 进行均值和标准差标准化（针对RGB通道）
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for c in range(image.shape[2]):
        image[..., c] = (image[..., c] - mean[c]) / std[c]
        
    # 转换为(C, H, W)格式
    image = image.transpose(2, 0, 1)  # Shape: (H, W, C) -> (C, H, W)

    # 增加batch维度 (1, C, H, W)
    image = image[None, ...]  # Add a new dimension for the batch (1, C, H, W)

    return image


# 后处理函数
def post_process(logits, cmap):
    print("Logits shape:", logits.shape)  # 打印 logits 的形状，调试时使用  维度为：(1, 21, 513, 513)

    # 检查 logits 的维度
    '''
    如果 logits 的维度是 (1, num_classes, H, W)，则使用 logits.max(1) 是正确的，但如果 logits 的维度是 (1, H, W)，那么直接使用 logits.max(1) 会出错。
    '''
    if len(logits.shape) == 3:  # 如果是 (H, W)，说明只有单个类别输出
        pred = logits.argmax(axis=0)  # 直接选择最大类别
    elif len(logits.shape) == 4:  # 如果是 (1, C, H, W)，选择最大类别的索引
        pred = logits.argmax(axis=1)[0]  # 去掉 batch 维度
    else:
        raise ValueError("Unexpected logits shape")

    # 将类别索引转换为颜色 
    pred = pred.astype(np.uint8)  # 转为 NumPy 数组
    colorized_preds = cmap[pred].astype('uint8')

    # 转为 PIL 图像并保存
    colorized_preds = Image.fromarray(colorized_preds)
    colorized_preds.save(os.path.join("samples/", 'res.png'))


# 使用onnxruntime进行推理的代码
def infer_onnx(image_path, onnx_model_path, output_path="samples/res.png"):
    # 加载onnx模型
    session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    # 读取并预处理图像
    image = cv2.imread(image_path)
    image = cv2.resize(image, (513, 513))  # 与训练时的图像大小一致
    image_input = preprocess(image)

    # 获取模型的输入输出名称
    input_name = session.get_inputs()[0].name  # 输入名称
    output_name = session.get_outputs()[0].name  # 输出名称

    # 执行推理
    logits = session.run([output_name], {input_name: image_input})[0]

    # 加载颜色映射
    cmap = voc_cmap()

    # 后处理和保存结果
    post_process(logits, cmap)

    print(f"Segmentation result saved to {output_path}")


# 在主程序中调用推理函数
if __name__ == "__main__":
    image_path = "samples/1_image.png"  # 输入图像路径
    onnx_model_path = "checkpoints/deeplabv3+.onnx"  # 导出的ONNX模型路径
    infer_onnx(image_path, onnx_model_path)
