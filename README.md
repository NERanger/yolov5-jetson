# Yolov5-jetson

yolov5 TensorRT implementation for running on Nvidia Jetson AGX Xavier with RealSense D435.

This repo uses [yolov5 release v3.0](https://github.com/ultralytics/yolov5/releases/tag/v3.0).

## Acknowledgement

This repo is a modified version of https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5.

The motivation is that the origin python implementation for yolov5 inference with TensorRT acceleration does not work on my Nvidia Jetson Xavier. Therefore, I use Pybind11 to add a Python interface for the C++ implementation.

## Config

The configuration approach is not well-designed, I will consider refactoring when I have time. Currently I just bare with the version from the original repo.

- Choose the model s/m/l/x by `NET` macro in [yolov5.cpp](src/yolov5.cpp)
- Input shape defined in [yololayer.h](plugin/yololayer.h)
- Number of classes defined in [yololayer.h](plugin/yololayer.h)
- FP16/FP32 can be selected by the macro in [yolov5.cpp](src/yolov5.cpp)
- GPU id can be selected by the macro in [yolov5.cpp](src/yolov5.cpp)
- NMS thresh in [yolov5.cpp](src/yolov5.cpp)
- BBox confidence thresh in [yolov5.cpp](src/yolov5.cpp)
- Batch size in [yolov5.cpp](src/yolov5.cpp)

## Usage (Yolov5s as an example)

1. Generate .wts from pytorch with .pt, or download .wts from model zoo
   * git clone source code of yolov5 v3.0
   * download https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt
   * copy scripts/gen_wts.py into ultralytics/yolov5
   * ensure the file name is yolov5s.pt and yolov5s.wts in gen_wts.py
   * go to this repo folder
   * Execute `python3 gen_wts.py`
   * a file 'yolov5s.wts' will be generated

2. Build this repo and run

   * Put yolov5s.wts into this repo folder

   * Update CLASS_NUM in yololayer.h if your model is trained on custom dataset

   * Execute the following bash commands

     ```bash
     mkdir build
     cd build
     cmake ..
     make
     // serialize model to plan file
     sudo ./yolov5 -s [.wts] [.engine] [s/m/l/x or c gd gw]
     // deserialize and run inference, the images in [image folder] will be processed.
     sudo ./yolov5 -d [.engine] [image folder]
     // For example yolov5s
     sudo ./yolov5 -s yolov5s.wts yolov5s.engine s
     sudo ./yolov5 -d yolov5s.engine ../samples
     // For example Custom model with depth_multiple=0.17, width_multiple=0.25 in yolov5.yaml
     sudo ./yolov5 -s yolov5_custom.wts yolov5.engine c 0.17 0.25
     sudo ./yolov5 -d yolov5.engine ../samples
     ```

3. check the images generated.

4. For inference with python, an example is given in [scripts/yolov5_infer.py](scripts/yolov5_infer.py)