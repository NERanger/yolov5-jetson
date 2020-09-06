#ifndef YOLOV5_PYWRAPPER_H
#define YOLOV5_PYWRAPPER_H

struct infer_result_t{
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;
    
};


#endif