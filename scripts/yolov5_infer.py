import cv2
import yolov5_module

img = cv2.imread('/home/bionicdl/repos/yolov5-jetson/sample/1.jpg', cv2.IMREAD_COLOR)

# print(img)

yolov5_module.init_inference('/home/bionicdl/repos/yolov5-jetson/build/yolov5s.engine')
res = yolov5_module.image_inference(img)

print(res)

yolov5_module.destory_inference()