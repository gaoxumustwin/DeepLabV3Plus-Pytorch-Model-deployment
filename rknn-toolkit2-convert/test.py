'''
DeepLabV3Plus-Pytorch rknn 转换推理
----------------------------------------------
@作者: gx
@邮箱: gaoxukkk@qq.com
@创建日期：2024年11月16日
'''

import os
import cv2
import numpy as np
from PIL import Image
from rknn.api import RKNN


ONNX_MODEL = 'model/deeplabv3+.onnx'
RKNN_MODEL = 'model/deeplabv3+.rknn'
IMG_PATH = "data/1_image.png"
DATASET = 'dataset.txt'

QUANTIZE_ON = True
IMG_SIZE = 513


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

def post_process(logits):

    print("Logits shape:", logits.shape) # Logits shape: (1, 21, 513, 513)

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
    colorized_preds.save(os.path.join("result/", 'res.png'))


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0.485*255, 0.456*255, 0.406*255]], std_values=[[0.229*255, 0.224*255, 0.225*255]], target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    # 转换为RGB
    image = img[..., ::-1] # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # 此时 的img为 (H, W, C)格式

    # img = img.transpose(2, 0, 1)  # 转换为(C, H, W)格式

    # Inference
    print('--> Running model')
    img2 = np.expand_dims(img, 0) # 增加batch维度 (1, H, W, C)
    logits = rknn.inference(inputs=[img2], data_format=['nhwc'])[0]
    print('done')

    # 加载颜色映射
    cmap = voc_cmap()
    
    post_process(logits)

    rknn.release()
