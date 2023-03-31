from ultralytics import YOLO
import cv2

model = YOLO('../YOLO_Weights/yolov8l.pt')
results = model('my_picture.jpg', show = True)
# results = model('driving.png', show = True)
# results = model('dogg.jpg', show = True)

cv2.waitKey(0)