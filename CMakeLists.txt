cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

find_package(CUDA REQUIRED)
find_package(realsense2 REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/plugin
    ${realsense2_INCLUDE_DIRS}
    )

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yolov5_plugin SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu ${PROJECT_SOURCE_DIR}/plugin/hardswish.cu)
target_link_libraries(yolov5_plugin nvinfer cudart)

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

add_executable(yolov5 ${PROJECT_SOURCE_DIR}/src/yolov5.cpp)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 yolov5_plugin)
target_link_libraries(yolov5 ${OpenCV_LIBS})

add_executable(yolov5-module ${PROJECT_SOURCE_DIR}/src/yolov5-module.cpp)
target_link_libraries(yolov5-module nvinfer)
target_link_libraries(yolov5-module cudart)
target_link_libraries(yolov5-module yolov5_plugin)
target_link_libraries(yolov5-module ${OpenCV_LIBS})

add_executable(yolov5-realsense ${PROJECT_SOURCE_DIR}/src/yolov5-realsense.cpp)
target_link_libraries(yolov5-realsense nvinfer)
target_link_libraries(yolov5-realsense cudart)
target_link_libraries(yolov5-realsense yolov5_plugin)
target_link_libraries(yolov5-realsense ${OpenCV_LIBS} ${realsense2_LIBRARY})

add_definitions(-O2 -pthread)

