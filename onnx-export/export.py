'''
DeepLabV3Plus-Pytorch ONNX导出
----------------------------------------------
@作者: gx
@邮箱: gaoxukkk@qq.com
@创建日期：2024年11月14日
'''

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import os

from network.modeling import deeplabv3plus_mobilenet

# BN层动量设置
def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

#  这个函数 voc_cmap 用于生成一个特定大小的颜色映射表（color map）。它的主要作用是为每个类别分配一个唯一的颜色，这在语义分割任务中非常常见，尤其是像 Pascal VOC 这样的数据集。
def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        '''
            这是一个辅助函数，目的是通过位操作从一个字节中提取指定位置的位值。byteval 是一个字节值（0-255），idx 是我们要提取的位的索引（从 0 到 7）。
            比如，如果 byteval = 5（二进制是 00000101），bitget(5, 0) 返回 1，bitget(5, 1) 返回 0。
        '''
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype) # cmap 是一个形状为 (N, 3) 的数组，用来存储 N 种颜色，每种颜色由 R、G、B 三个分量表示。
    
    # 对于每个颜色 i，它使用一个 8 位整数（i）来生成对应的 RGB 值。
    # 每个颜色的 R、G、B 值是通过将整数 i 的不同位（从低位到高位）映射到 RGB 的不同分量中来得到的。
    # 256个颜色组合 对应 2的8次幂
    for i in range(N):   
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    # 不将颜色进行归一化
    cmap = cmap/255 if normalized else cmap
    return cmap


class DeepLabV3Plus(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # voc数据集
        self.num_classes = 21
        self.output_stride = 16

        self.model = deeplabv3plus_mobilenet(num_classes = self.num_classes, output_stride = self.output_stride)
        # 修改模型中的BN层的动量
        set_bn_momentum(self.model.backbone, momentum=0.01)
        
        # 加载模型
        state_dict = torch.load("checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth", map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict["model_state"])
    
    def forward(self, x):

        # 获取模型的原始输出 (logits)
        logits = self.model(x)
        
        # 不要在这里进行 numpy 转换
        return logits

        
device = "cpu"
model = DeepLabV3Plus().eval().to(device)


# ------- 数据导入 -------
image = cv2.imread("samples/1_image.png")
image = cv2.resize(image, (513, 513))


# ------- 预处理 -------
# To RGB
image = image[..., ::-1]  # 是一种toRGB的方法 

#  Normalize the image (scale pixel values to [0, 1])
image = (image / 255.0).astype(np.float32)

# Normalize with given mean and std for each channel (RGB)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
for c in range(image.shape[2]):  # Iterate over the RGB channels
    image[..., c] = (image[..., c] - mean[c]) / std[c]  # Standardize each channel
    
# Convert image to (C, H, W) format (channels first)
image = image.transpose(2, 0, 1)  # Shape: (H, W, C) -> (C, H, W)

# Add batch dimension (1, C, H, W)
image = image[None, ...]  # Add a new dimension for the batch (1, C, H, W)

# Convert to PyTorch tensor
image = torch.from_numpy(image).to(device)


# ------- 后处理 -------
cmap = voc_cmap()
with torch.no_grad():
    logits = model(image)

    # 这里才可以进行 max 和 numpy 转换
    pred = logits.max(1)[1].cpu().numpy()[0]  # 获取预测的类别索引，并转为 NumPy 数组

    # 进行颜色映射
    colorized_preds = cmap[pred].astype('uint8')
    colorized_preds = Image.fromarray(colorized_preds)
    colorized_preds.save(os.path.join("samples/", 'res.png'))


# ------- 导出 ONNX 模型 -------
import onnx

input_layer_names = ["images"]
output_layer_names = ["output"]
model_path = "checkpoints/deeplabv3+.onnx"

# 导出模型
print(f'Starting export with onnx {onnx.__version__}.')
torch.onnx.export(
    model,  # 模型对象
    (image,),  # 模型的输入（输入数据）
    f=model_path,  # 保存的 ONNX 文件路径
    verbose=False,  # 是否打印详细信息
    opset_version=12,  # 使用 opset 版本 12
    do_constant_folding=True,  # 是否进行常量折叠优化
    input_names=input_layer_names,  # 输入层的名字
    output_names=output_layer_names,  # 输出层的名字
    dynamic_axes=None  # 是否有动态维度
)

# ------- 检查 ONNX 模型 -------
model_onnx = onnx.load(model_path)  # 加载 ONNX 模型
onnx.checker.check_model(model_onnx)  # 检查 ONNX 模型

# ------- 简化 ONNX -------
import onnxsim
print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
model_onnx, check = onnxsim.simplify(
    model_onnx,
    dynamic_input_shape=False,
    input_shapes=None)
assert check, 'assert check failed'
onnx.save(model_onnx, model_path)

print(f'Onnx model saved as {model_path}')


print("done")