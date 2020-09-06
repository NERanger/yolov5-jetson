cmake_minimum_required(VERSION 3.4...3.18)

project(yolov5)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

find_package(CUDA REQUIRED)
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/plugin
    ${realsense2_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
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

add_subdirectory(pybind11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yolov5_plugin SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu ${PROJECT_SOURCE_DIR}/plugin/hardswish.cu)
target_link_libraries(yolov5_plugin nvinfer cudart)

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)

add_executable(yolov5 
    ${PROJECT_SOURCE_DIR}/src/yolov5.cpp
    ${PROJECT_SOURCE_DIR}/src/common.cpp
)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 yolov5_plugin)
target_link_libraries(yolov5 ${OpenCV_LIBS})

add_executable(yolov5-realsense
    ${PROJECT_SOURCE_DIR}/src/yolov5-realsense.cpp
    ${PROJECT_SOURCE_DIR}/src/common.cpp
)
target_link_libraries(yolov5-realsense nvinfer)
target_link_libraries(yolov5-realsense cudart)
target_link_libraries(yolov5-realsense yolov5_plugin)
target_link_libraries(yolov5-realsense ${OpenCV_LIBS} ${realsense2_LIBRARY})

add_definitions(-O2 -pthread)

pybind11_add_module(yolov5_module 
    ${PROJECT_SOURCE_DIR}/src/yolov5-pywrapper.cpp
    ${PROJECT_SOURCE_DIR}/src/yolov5-module.cpp
    ${PROJECT_SOURCE_DIR}/src/common.cpp
)

target_link_libraries(yolov5_module PRIVATE nvinfer)
target_link_libraries(yolov5_module PRIVATE cudart)
target_link_libraries(yolov5_module PRIVATE yolov5_plugin)
target_link_libraries(yolov5_module PRIVATE ${OpenCV_LIBS})
