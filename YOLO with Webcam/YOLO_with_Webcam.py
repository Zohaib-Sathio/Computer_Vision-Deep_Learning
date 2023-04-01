from ultralytics import YOLO
import cv2
import cvzone
import time
import math

# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

model = YOLO('../YOLO_Weights/yolov8n.pt')

# cap = cv2.VideoCapture('../Video Files/new-york-city-1044.mp4')
# cap = cv2.VideoCapture('../Video Files/car-2165.mp4')
cap = cv2.VideoCapture('../Video Files/cat_and_dog.mp4')
# cap = cv2.VideoCapture('../Video Files/street-38590.mp4')

classNames = ['person', 'bicycle', 'car', 'motorcycle',
              'airplane','bus','train','truck',   'boat', 'traffic light','fire hydrant', 'stop sign',
              'parking meter','bench', 'bird','cat', 'dog','horse','sheep',  'cow',  'elephant',  'bear',   'zebra',
              'giraffe',   'backpack','umbrella',   'handbag', 'tie',  'suitcase',  'frisbee',
              'skis',  'snowboard',   'sports ball',  'kite','baseball bat',   'baseball glove',
              'skateboard', 'surfboard', 'tennis racket',  'bottle', 'wine glass', 'cup',  'fork','knife',
              'spoon',   'bowl', 'banana',  'apple', 'sandwich',  'orange', 'broccoli', 'carrot',  'hot dog', 'pizza',
              'donut',  'cake', 'chair',  'couch','potted plant',  'bed',  'dining table',
              'toilet', 'tv','laptop', 'mouse', 'remote',  'keyboard', 'cell phone',  'microwave',  'oven',  'toaster',   'sink',
              'refrigerator',   'book', 'clock', 'vase', 'scissors',  'teddy bear', 'hair drier', 'toothbrush']

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1,y1,x2,y2 = box.xyxy[0]

            # Confidence
            conf = math.ceil(box.conf[0]*100) / 100

            # Class name
            cls = box.cls[0]

            if conf > 0.4:
                cvzone.cornerRect(img, (int(x1),int(y1), int(x2)-int(x1), int(y2)-int(y1)),15, 3)
                cvzone.putTextRect(img, f'{classNames[int(cls)]} {conf}', (int(x1),int(y1)),1,1, offset=7)

    cv2.imshow("Detection Frame", img)
    cv2.waitKey(1)
