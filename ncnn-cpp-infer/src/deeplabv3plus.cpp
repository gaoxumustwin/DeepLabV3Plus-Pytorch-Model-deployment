'''
DeepLabV3Plus-Pytorch ncnn c++推理
----------------------------------------------
@作者: gx
@邮箱: gaoxukkk@qq.com
@创建日期：2024年11月15日
'''


#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <iostream>



// 定义颜色映射生成函数，模拟 Python 中的 voc_cmap
cv::Mat create_voc_color_map()
{
    int N = 21;
    cv::Mat cmap(N, 1, CV_8UC3);
    for (int i = 0; i < N; i++) {
        int r = 0, g = 0, b = 0;
        int c = i;
        for (int j = 0; j < 8; j++) {
            // opencv 是BGR的色彩空间  
            r = r | ((c & (1 << 0)) >> 0) << (7 - j);
            g = g | ((c & (1 << 1)) >> 1) << (7 - j);
            b = b | ((c & (1 << 2)) >> 2) << (7 - j);

            c = c >> 3;
        }
        // cv::Vec3b的通道顺序是r, g, b, 而opencv的通道顺序是BGR，所以我们在上面将r和b交换了顺序   所以这里的

        /*
        i: 0 b：0 g：0 r：0
        i: 1 b：0 g：0 r：128 此时cv::Vec3b(0, 0, 128); 红色
        */
        // 注意opencv的色彩空间为b g r
        cmap.at<cv::Vec3b>(i) = cv::Vec3b(b, g, r);
        /*
        这些是VOC的颜色映射，按照类别编号赋值
        cmap.at<cv::Vec3b>(0) = cv::Vec3b(0, 0, 0);       // 背景
        cmap.at<cv::Vec3b>(1) = cv::Vec3b(0, 0, 128);     // 类别1 -> Red
        cmap.at<cv::Vec3b>(2) = cv::Vec3b(0, 128, 0);     // 类别2 -> Green
        cmap.at<cv::Vec3b>(3) = cv::Vec3b(128, 0, 0);     // 类别3 -> Blue
        cmap.at<cv::Vec3b>(4) = cv::Vec3b(0, 128, 128);   // 类别4
        cmap.at<cv::Vec3b>(5) = cv::Vec3b(128, 0, 128);   // 类别5
        cmap.at<cv::Vec3b>(6) = cv::Vec3b(128, 128, 0);   // 类别6
        cmap.at<cv::Vec3b>(7) = cv::Vec3b(192, 128, 128); // 类别7
        cmap.at<cv::Vec3b>(8) = cv::Vec3b(128, 192, 0);   // 类别8
        cmap.at<cv::Vec3b>(9) = cv::Vec3b(255, 255, 0);   // 类别9
        cmap.at<cv::Vec3b>(10) = cv::Vec3b(255, 0, 255);  // 类别10
        cmap.at<cv::Vec3b>(11) = cv::Vec3b(255, 255, 255); // 类别11
        cmap.at<cv::Vec3b>(12) = cv::Vec3b(255, 192, 0);  // 类别12
        cmap.at<cv::Vec3b>(13) = cv::Vec3b(255, 128, 255);  // 类别13
        cmap.at<cv::Vec3b>(14) = cv::Vec3b(192, 255, 128); // 类别14
        cmap.at<cv::Vec3b>(15) = cv::Vec3b(0, 255, 255);  // 类别15
        cmap.at<cv::Vec3b>(16) = cv::Vec3b(0, 128, 255);  // 类别16
        cmap.at<cv::Vec3b>(17) = cv::Vec3b(255, 0, 0);   // 类别17
        cmap.at<cv::Vec3b>(18) = cv::Vec3b(192, 0, 255);  // 类别18
        cmap.at<cv::Vec3b>(19) = cv::Vec3b(255, 128, 128); // 类别19
        cmap.at<cv::Vec3b>(20) = cv::Vec3b(128, 0, 255); // 类别20        
        */

    }

    return cmap;
}

static int seg_deeplabv3plus(const cv::Mat& bgr)
{
    ncnn::Net deeplabv3plus;

    deeplabv3plus.opt.use_vulkan_compute = true;

    // 加载模型的参数和数据
    if (deeplabv3plus.load_param("../checkpoints/deeplabv3+.param"))
    {
        fprintf(stderr, "Load param failed\n");
        return -1;
    }
    if (deeplabv3plus.load_model("../checkpoints/deeplabv3+.bin"))
    {
        fprintf(stderr, "Load model failed\n");
        return -1;
    }

    // OpenCV读取图片是BGR格式，需要转换为RGB格式
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 513, 513);

    // 图像归一化处理
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1 / (0.229f * 255.f), 1 / (0.224f * 255.f), 1 / (0.225f * 255.f)};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // 创建提取器
    ncnn::Extractor ex = deeplabv3plus.create_extractor();
    ex.input("images", in);

    ncnn::Mat out;
    ex.extract("output", out);  // 提取推理结果

    // 查看 ncnn::Mat 的维度
    // printf("Output dimensions:\n");
    // printf("Height: %d\n", out.h);  // 输出高度 513
    // printf("Width: %d\n", out.w);   // 输出宽度 513
    // printf("Channels: %d\n", out.c);  // 输出通道数 21

    // 后处理：将 ncnn::Mat 转为 OpenCV 格式的结果图像
    // 获取输出的维度：21个类别，513x513的分辨率
    int height = out.h;
    int width = out.w;
    int channels = out.c;  // 应该是 21

    // 解析模型输出，选择每个像素的最大值作为类别
    std::vector<int> pred(height * width);
    for (int i = 0; i < height * width; i++) {
        float max_prob = -1;
        int max_label = 0;
        for (int j = 0; j < channels; j++) {
            float prob = out.channel(j)[i];
            if (prob > max_prob) {
                max_prob = prob;
                max_label = j;
            }
        }
        pred[i] = max_label;  // 将每个像素的类别索引保存
    }

    // 创建一个与输出相同大小的 RGB 图像（彩色化的语义分割结果）
    cv::Mat colorized(height, width, CV_8UC3);
    cv::Mat cmap = create_voc_color_map();  // 获取VOC颜色映射

    // 将预测结果与颜色映射关联
    for (int i = 0; i < height * width; i++) {
        int label = pred[i];
        if (label >= 0 && label < 21) { // 21个类
            colorized.at<cv::Vec3b>(i / width, i % width) = cmap.at<cv::Vec3b>(label);
            // printf("label:%d\n", label); // 只有0和1
        }
        else {
            colorized.at<cv::Vec3b>(i / width, i % width) = cv::Vec3b(0, 0, 0); // 背景处理
        }
    }

    // 保存结果
    cv::imwrite("../samples/result.png", colorized);
    printf("Segmentation result saved to samples/result.png\n");


    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    // 使用 OpenCV 读取图像
    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    seg_deeplabv3plus(m);

    return 0;
}
