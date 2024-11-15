# DeepLabV3Plus-Pytorch-ONNX-export
DeepLabV3Plus Pytorch onnx export tutorial, with implementation of pre-processing and post-processing in the code

## Address using DeepLabV3Plus

https://github.com/VainF/DeepLabV3Plus-Pytorch

## How to Use

step1: Clone repository

```sh
git clone https://github.com/VainF/ 
```

step2: Configure environment

My GPU device is 3060, my Python environment is 3.8, and my CUDA version is 11.7. Other Python libraries can be referred to as follows

```sh
certifi==2024.8.30
charset-normalizer==3.4.0
contourpy==1.1.1
cycler==0.12.1
fonttools==4.54.1
idna==3.10
importlib_resources==6.4.5
joblib==1.4.2
jsonpatch==1.33
jsonpointer==3.0.0
kiwisolver==1.4.7
matplotlib==3.7.5
networkx==3.1
numpy==1.24.4
packaging==24.2
pillow==10.4.0
pyparsing==3.1.4
python-dateutil==2.9.0.post0
requests==2.32.3
scikit-learn==1.3.2
scipy==1.10.1
six==1.16.0
threadpoolctl==3.5.0
# torch @ torch-1.13.0%2Bcu117-cp38-cp38-linux_x86_64.whl  本地安装
# torchvision @ torchvision-0.14.0%2Bcu117-cp38-cp38-linux_x86_64.whl 本地安装
tornado==6.4.1
tqdm==4.67.0
typing_extensions==4.12.2
urllib3==2.2.3
visdom==0.2.4
websocket-client==1.8.0
zipp==3.20.2
```

Just configure according to the requirements. txt file

step3: testing environment

```sh
# workspace  DeepLabV3Plus-Pytorch
mkdir checkpoints # Download and train the weight model to the file below （Refer to the README.md of DeepLabV3Plus Pytorch）
python predict.py --input samples/23_image.png --dataset voc --model deeplabv3plus_mobilenet --ckpt /mnt/d/Users/XueLi_G/Desktop/DeepLabV3Plus/DeepLabV3Plus-Pytorch/checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth --save_val_results_to test_results
```

I used an absolute path in my -- ckpt because I encountered an issue with specifying a relative path and the model couldn't pass it in

step4: Move the export file to DeepLabV3Plus-Pytorch（Same level as main.by） and execute it using Python

```sh
python export.py
```

After execution, deeplabv3+. onnx will be generated under the checkpoints file and res.png will be generated under the samples file
