# DeepLabV3Plus-Pytorch-NCNN-C++-infer
## How to Use

step1: delete build and bin

```sh
# workspace ncnn-cpp-infer
rm -r build
rm -r bin
```

step2: Create a build folder and enter

```sh
# workspace ncnn-cpp-infer
mkdir build
cd build
```

step3: Compile and Build

```sh
# build
cmake ..
make
```

A bin folder will be generated under ncnn cpp infer, containing an executable file

step4: test run 

```sh
# build
cd ../bin
# bin
./deeplabv3plus  ../samples/1_image.png
```

Generate a res.png file under the samples folder as a result
