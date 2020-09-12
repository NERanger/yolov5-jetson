#ifndef YOLOV5_PYWRAPPER_H
#define YOLOV5_PYWRAPPER_H

#include <array>

struct infer_result_t{
    //center_x center_y w h
    std::array<float, 4> bbox;
    float conf;  // bbox_conf * cls_conf
    float class_id;
};


#endif