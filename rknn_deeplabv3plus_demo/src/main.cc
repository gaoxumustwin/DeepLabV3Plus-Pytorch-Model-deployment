'''
DeepLabV3Plus-Pytorch rknpu c++推理
----------------------------------------------
@作者: gx
@邮箱: gaoxukkk@qq.com
@创建日期：2024年11月17日
'''

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rknn_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

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

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE* fp = fopen(filename, "rb");
  if (fp == nullptr) {
    printf("fopen %s fail!\n", filename);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  int            model_len = ftell(fp);
  unsigned char* model     = (unsigned char*)malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if (model_len != fread(model, 1, model_len, fp)) {
    printf("fread %s fail!\n", filename);
    free(model);
    return NULL;
  }
  *model_size = model_len;
  if (fp) {
    fclose(fp);
  }
  return model;
}


static void Post_Process(float* output_data)
{
    // 后处理：将 ncnn::Mat 转为 OpenCV 格式的结果图像
    // 获取输出的维度：21个类别，513x513的分辨率
    int height = 513;
    int width = 513;
    int classes = 21;  // 应该是 21

    // 解析模型输出，选择每个像素的最大值作为类别
    std::vector<int> pred(height * width);
    for (int i = 0; i < height * width; i++) { // 遍历每个像素
        float max_prob = -1;
        int max_label = 0;
        for (int j = 0; j < classes; j++) { // 遍历21个类别
            float prob = output_data[j * height * width + i]; // RKNN 输出的格式通常是按类别排布
            // float prob = out.channel(j)[i]; // ncnn的推理方式
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
    cv::imwrite("model/result.png", colorized);
    printf("Segmentation result saved to samples/result.png\n");
  }


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{ 
  // 设置输入的尺寸
  const int MODEL_IN_WIDTH    = 513;
  const int MODEL_IN_HEIGHT   = 513;
  const int MODEL_IN_CHANNELS = 3;

  // 设备上下文
  rknn_context ctx = 0;
  int            ret;
  int            model_len = 0;
  unsigned char* model;

  // 模型参数和图像参数
  const char* model_path = argv[1];
  const char* img_path   = argv[2];

  // 控制台必须有三个参数指定
  if (argc != 3) {
    printf("Usage: %s <rknn model> <image_path> \n", argv[0]);
    return -1;
  }

  // Load image
  cv::Mat orig_img = imread(img_path, cv::IMREAD_COLOR);
  if (!orig_img.data) {
    printf("cv::imread %s fail!\n", img_path);
    return -1;
  }

  // 色彩空间转换
  cv::Mat orig_img_rgb;
  cv::cvtColor(orig_img, orig_img_rgb, cv::COLOR_BGR2RGB);

  cv::Mat img = orig_img_rgb.clone();
  if (orig_img.cols != MODEL_IN_WIDTH || orig_img.rows != MODEL_IN_HEIGHT) {
    printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
    cv::resize(orig_img_rgb, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), 0, 0, cv::INTER_LINEAR);
  }

  // Load RKNN Model
  model = load_model(model_path, &model_len);
  ret   = rknn_init(&ctx, model, model_len, 0, NULL);
  if (ret < 0) {
    printf("rknn_init fail! ret=%d\n", ret);
    return -1;
  }

  // 查询模型的输入 和 输出 信息
  rknn_input_output_num io_num; // 表示输入输出tensor个数  属性n_input为输入tensor个数  属性n_output为输出tensor个数。
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  // 打印模型的输入输出的tensor个数
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  printf("input tensors:\n");
  rknn_tensor_attr input_attrs[io_num.n_input]; // rknn_tensor_attr表示模型的tensor的属性
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) { // 遍历每个输入
    input_attrs[i].index = i; // 设置输入的索引
    // rknn_query可以查询获取到模型输入输出信息、逐层运行时间、模型推理的总时间、SDK版本、内存占用信息、用户自定义字符串等信息。  
    ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) { // RKNN_SUCC(0) 是执行成功
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    // 编写的打印函数  将 rknn_query 查询到的信息打印出来
    dump_tensor_attr(&(input_attrs[i]));
  }

  // 和输入同理
  printf("output tensors:\n");
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(output_attrs[i]));
  }

  // 设置输入
  rknn_input inputs[1]; // 结构体rknn_input表示模型的一个数据输入 用来作为参数传入给rknn_inputs_set函数。
  memset(inputs, 0, sizeof(inputs)); // 将 inputs 指向的整个内存块中的所有字节都设置为 0
  inputs[0].index = 0; // 该输入的索引位置。
  inputs[0].type  = RKNN_TENSOR_UINT8; // 输入数据的类型。
  inputs[0].size  = img.cols * img.rows * img.channels() * sizeof(uint8_t); // 输入数据所占内存大小。 大小为输入数据的宽*高*通道数
  inputs[0].fmt   = RKNN_TENSOR_NHWC; //输入数据的格式。 
  inputs[0].buf   = img.data; // 输入数据的指针

  // 上面设置的是rknn_input结构体对象  将其传递给 rknn_inputs_set函数 进行模型的输入
  ret = rknn_inputs_set(ctx, io_num.n_input, inputs); // 通过rknn_inputs_set函数可以设置模型的输入数据。该函数能够支持多个输入，其中每个输入是rknn_input结构体对象
  if (ret < 0) {
    printf("rknn_input_set fail! ret=%d\n", ret);
    return -1;
  }

  // Run
  printf("rknn_run\n");
  ret = rknn_run(ctx, nullptr); // rknn_run函数将执行一次模型推理，调用之前需要先通过rknn_inputs_set函数或者零拷贝的接口设置输入数据。
  if (ret < 0) {
    printf("rknn_run fail! ret=%d\n", ret);
    return -1;
  }

  // Get Output
  rknn_output outputs[1]; // 结构体rknn_output表示模型的一个数据输出，用来作为参数传入给rknn_outputs_get函数，在函数执行后，结构体对象将会被赋值。
  memset(outputs, 0, sizeof(outputs));
  outputs[0].want_float = 1; // 标识是否需要将输出数据转为float类型输出，该字段由用户设置。
  // rknn_outputs_get函数可以获取模型推理的输出数据。该函数能够一次获取多个输出数据。其中每个输出是rknn_output结构体对象，在函数调用之前需要依次创建并设置每个rknn_output对象。
  ret                   = rknn_outputs_get(ctx, 1, outputs, NULL); // 1 为 输出数据个数
  if (ret < 0) {
    printf("rknn_outputs_get fail! ret=%d\n", ret);
    return -1;
  }

  // outputs即为模型输出

  // Post Process
  for (int i = 0; i < io_num.n_output; i++) { // 遍历模型的输出 
    float*   buffer = (float*)outputs[i].buf;


    Post_Process(buffer);
  }

  // Release rknn_outputs
  rknn_outputs_release(ctx, 1, outputs); // 对于输出数据的buffer存放可以采用两种方式：一种是用户自行申请和释放，此时rknn_output对象的is_prealloc需要设置为1，并且将buf指针指向用户申请的buffer；

  // Release
  if (ctx > 0)
  {
    rknn_destroy(ctx);
  }
  if (model) {
    free(model);
  }
  return 0;
}
