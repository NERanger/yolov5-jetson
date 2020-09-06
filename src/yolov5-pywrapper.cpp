#include <algorithm>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl_bind.h"
#include "pybind11/stl.h"

#include "yolov5-module.hpp"
#include "yolov5-pywrapper.hpp"

PYBIND11_MAKE_OPAQUE(std::vector<infer_result_t>)

namespace py = pybind11;

cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<uint8_t>& input) {

    if (input.ndim() != 3)
        throw std::runtime_error("3-channel image must be 3 dims ");

    py::buffer_info buf = input.request();

    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (uint8_t*)buf.ptr);

    return mat;
}

void py_init_inference(const std::string& engine_path){
    initInference(engine_path);
}

std::vector<infer_result_t> py_image_inference(py::array_t<uint8_t>& input){

    cv::Mat img = numpy_uint8_3c_to_cv_mat(input);
    std::vector<Yolo::Detection> res =  imgInference(img);

    std::vector<infer_result_t> py_res;

    size_t res_size = res.size();
    for(size_t i = 0; i < res_size; i++){
        infer_result_t r;
        std::copy(std::begin(res[i].bbox), std::end(res[i].bbox), std::begin(r.bbox));
        r.class_id = res[i].class_id;
        r.conf = res[i].conf;
    }

    return py_res;
}

void py_destory_inference(){
    destoryInference();
}

PYBIND11_MODULE(yolov5_module, m){
    py::class_<infer_result_t>(m, "infer_result")
        .def(py::init<>())
        .def_readonly("bbox", &infer_result_t::bbox)
        .def_readonly("conf", &infer_result_t::conf)
        .def_readonly("classid", &infer_result_t::class_id);

    py::bind_vector<std::vector<infer_result_t>>(m, "vec_infer_result");

    m.def("init_inference", &py_init_inference);
    m.def("image_inference", &py_image_inference);
    m.def("destory_inference", &py_destory_inference);
}