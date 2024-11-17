# Aarch64 Linux 示例

## 安装RKNPU2的环境

- 确认 RKNPU2 驱动版本

可以在板端执行以下命令查询 RKNPU2 驱动版本：

```sh
orangepi@orangepi5b:~$ dmesg | grep -i rknpu
[    8.706925] RKNPU fdab0000.npu: Adding to iommu group 0
[    8.707062] RKNPU fdab0000.npu: RKNPU: rknpu iommu is enabled, using iommu mode
[    8.707152] RKNPU fdab0000.npu: Looking up rknpu-supply from device tree
[    8.707731] RKNPU fdab0000.npu: Looking up mem-supply from device tree
[    8.708383] RKNPU fdab0000.npu: can't request region for resource [mem 0xfdab0000-0xfdabffff]
[    8.708404] RKNPU fdab0000.npu: can't request region for resource [mem 0xfdac0000-0xfdacffff]
[    8.708416] RKNPU fdab0000.npu: can't request region for resource [mem 0xfdad0000-0xfdadffff]
[    8.708963] [drm] Initialized rknpu 0.9.6 20240322 for fdab0000.npu on minor 0
[    8.709776] RKNPU fdab0000.npu: Looking up rknpu-supply from device tree
[    8.710264] RKNPU fdab0000.npu: Looking up mem-supply from device tree
[    8.710823] RKNPU fdab0000.npu: Looking up rknpu-supply from device tree
[    8.712415] RKNPU fdab0000.npu: RKNPU: bin=0
[    8.712673] RKNPU fdab0000.npu: leakage=9
[    8.712712] RKNPU fdab0000.npu: Looking up rknpu-supply from device tree
[    8.712742] debugfs: Directory 'fdab0000.npu-rknpu' with parent 'vdd_npu_s0' already present!
[    8.719641] RKNPU fdab0000.npu: pvtm=847
[    8.724112] RKNPU fdab0000.npu: pvtm-volt-sel=2
[    8.724807] RKNPU fdab0000.npu: avs=0
[    8.724936] RKNPU fdab0000.npu: l=10000 h=85000 hyst=5000 l_limit=0 h_limit=800000000 h_table=0
[    8.736936] RKNPU fdab0000.npu: failed to find power_model node
[    8.736948] RKNPU fdab0000.npu: RKNPU: failed to initialize power model
[    8.736955] RKNPU fdab0000.npu: RKNPU: failed to get dynamic-coefficient
```

如上面所示，当前 RKNPU2 驱动版本为 0.9.6（[    8.708963] [drm] Initialized rknpu 0.9.6 20240322 for fdab0000.npu on minor 0）。

- 安装/更新 RKNPU2 环境

不同的板端系统需要安装不同的 RKNPU2 环境，下面分别介绍各自的安装方法。注：如果已经安装版本一致 RKNPU2 环境，则这里可以跳过。

板端为 Linux 系统，进入 rknpu2 目录，使用 adb 工具将相应的 rknn_server 和 librknnrt.so 推送至板端，然后启动 rknn_server，参考命令如下：

```sh
# 进入 rknpu2 目录   (本地PC机执行)
cd Projects/rknn-toolkit2/rknpu2

# 推送 rknn_server 到板端 (本地PC机执行)
# 注：在64位Linux系统中，BOARD_ARCH对应aarch64目录，在32位系统，对应armhf目录。
adb push runtime/Linux/rknn_server/${BOARD_ARCH}/usr/bin/* /usr/bin
# 例如我的是:adb push runtime/Linux/rknn_server/aarch64/usr/bin/* /usr/bin

# 推送 librknnrt.so
adb push runtime/Linux/librknn_api/${BOARD_ARCH}/librknnrt.so /usr/lib
# 例如我的是:adb push runtime/Linux/librknn_api/aarch64/usr/bin/* /usr/bin

# 进入板端
adb shell # (本地PC机执行，执行完后操控开发板)

# 赋予可执行权限 （开发板执行）
sudo chmod +x /usr/bin/rknn_server
sudo chmod +x /usr/bin/start_rknn.sh
sudo chmod +x /usr/bin/restart_rknn.sh

# 重启 rknn_server 服务  （开发板执行）
restart_rknn.sh
```

- 检查 RKNPU2 环境是否安装

RKNN-Toolkit2 的连板调试功能要求板端已安装 RKNPU2 环境，并且启动 rknn_server 服务。以下是RKNPU2 环境中的两个基本概念：

- RKNN Server：一个运行在开发板上的后台代理服务。该服务的主要功能是调用板端 Runtime 对应的接口处理计算机通过USB传输过来的数据，并将处理结果返回给计算机。
- RKNPU2 Runtime 库（librknnrt.so）：主要职责是负责在系统中加载 RKNN 模型，并通过调用专用的神经处理单元（NPU）执行 RKNN 模型的推理操作。

如果板端没有安装 RKNN Server 和 Runtime 库，或者 RKNN Server 和 Runtime 库的版本不一致，都需要重新安装 RKNPU2 环境。（注意：1. 若使用动态维度输入的 RKNN 模型，则要求 RKNN Server 和Runtime 库版本 >= 1.5.0。2. 要保证 RKNN Server 、Runtime 库的版本、RKNN-Toolkit2 的版本是一致的，建议都安装最新的版本）

通常情况下，开发板默认已经安装版本一致的 RKNPU2 环境，可以通过下面命令确认，板端为 Linux 系统:

- 检查 RKNPU2 环境是否安装

如果能够启动 rknn_server 服务，则代表板端已经安装 RKNPU2 环境。

```sh
# 进入板端
adb shell

# 启动 rknn_server
restart_rknn.sh
# 出现以下信息，则代表启动rknn_server服务成功，即已经安装RKNPU2环境
start rknn server, version:2.3.0 (e80ac5c build@2024-11-07T12:52:53)
I NPUTransfer(3345): Starting NPU Transfer Server, Transfer version 2.2.2 (@2024-06-18T03:50:51)
```

- 检查版本是否一致

```sh
# 查询rknn_server版本
strings /usr/bin/rknn_server | grep -i "rknn_server version" 
# 我的是：rknn_server version: 2.3.0 (e80ac5c build@2024-11-07T12:52:53)

# 查询librknnrt.so库版本
strings /usr/lib/librknnrt.so | grep -i "librknnrt version"
# 我的是:librknnrt version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
```

两个版本都是2.3.0，所以检查版本是一致的



## gcc交叉编译器的下载和安装方法

- 开发板为： Linux 系统的开发板

- GCC 下载地址

板端为 64 位系统的交叉编译器下载地址：https://releases.linaro.org/components/toolchain/binaries/6.3-2017.05/aarch64-linux-gnu/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz

- 解压软件包

建议将 GCC 软件包解压到 Projects 的文件夹中。

```
tar -xvf gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz
```

以板端为 64 位系统的 GCC 软件包为例，存放位置如下：

```sh
Projects
├── rknn-toolkit2
├── rknn_model_zoo
└── gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu # 此路径在后面编译RKNN C Demo时会用到
```

此时， GCC 编译器的路径是 Projects/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu

## 编译

指定编译器

```sh
export GCC_COMPILER=/mnt/e/ubuntu20.04/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu
```

然后执行如下命令：

```sh
./build-linux.sh -t <target> -a <arch> -b <build_type>]

# 例如: 
./build-linux.sh -t rk3588 -a aarch64 -b Release
```

## 安装

将 install/rknn_api_demo_Linux 拷贝到设备上。

- 如果使用Rockchip的EVB板，可以使用以下命令：

连接设备并将程序和模型传输到`/userdata`

```sh
adb push install/rknn_mobilenet_demo_Linux /userdata/

install/
└── rknn_mobilenet_demo_Linux
    ├── lib
    │   └── librknnrt.so
    ├── model
    │   ├── RK3588
    │   │   └── mobilenet_v1.rknn
    │   ├── cat_224x224.jpg
    │   └── dog_224x224.jpg
    └── rknn_mobilenet_demo
```

将install下面的rknn_mobilenet_demo_Linux文件上传到开发板的某个问价下面，例如：/userdata/

## 运行


```
adb shell
cd /userdata/rknn_mobilenet_demo_Linux/
```

```sh
export LD_LIBRARY_PATH=./lib
./rknn_mobilenet_demo model/<TARGET_PLATFORM>/mobilenet_v1.rknn model/dog_224x224.jpg # TARGET_PLATFORM> 表示RK3566_RK3568、RK3562、RK3576或RK3588。

# 例如：./rknn_mobilenet_demo ./model/RK3588/mobilenet_v1.rknn ./model/cat_224x224.jpg 
```

