#ifndef YOLOV5_MODULE_H
#define YOLOV5_MODULE_H

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

#define NET s  // s m l x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)
#define STR1(x) #x
#define STR2(x) STR1(x)

#include <opencv2/core/core.hpp>
#include <cuda_runtime_api.h>

#include "logging.h"
#include "common.hpp"

ICudaEngine* createEngine_s(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
ICudaEngine* createEngine_m(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
ICudaEngine* createEngine_l(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
ICudaEngine* createEngine_x(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);
void doInference(IExecutionContext& context, float* input, float* output, int batchSize);

void initInference(const std::string& engine_path);
std::vector<Yolo::Detection> imgInference(cv::Mat& img);
void destoryInference();

#endif