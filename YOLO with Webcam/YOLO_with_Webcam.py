from ultralytics import YOLO
import cv2
import cvzone
import time
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO('../YOLO_Weights/yolov8n.pt')

classNames = ['person', 'bicycle', 'car', 'motorcycle',
              'airplane','bus','train','truck',   'boat', 'traffic light',
              'fire hydrant', 'stop sign',
              'parking meter','bench', 'bird','cat',
              'dog','horse','sheep',  'cow',  'elephant',  'bear',   'zebra',
              'giraffe',   'backpack',
              'umbrella',   'handbag', 'tie',  'suitcase',  'frisbee',
              'skis',  'snowboard',   'sports ball',  'kite',
              'baseball bat',   'baseball glove',
              'skateboard', 'surfboard', 'tennis racket',  'bottle',
              'wine glass', 'cup',  'fork','knife',
              'spoon',   'bowl', 'banana',  'apple', 'sandwich',  'orange',
              'broccoli', 'carrot',  'hot dog', 'pizza',
              'donut',  'cake', 'chair',  'couch',
              'potted plant',  'bed',  'dining table',
              'toilet', 'tv','laptop', 'mouse', 'remote',  'keyboard',
              'cell phone',  'microwave',  'oven',  'toaster',   'sink',
              'refrigerator',   'book', 'clock', 'vase',
              'scissors',  'teddy bear', 'hair drier', 'toothbrush']

fTime = 0
cTime = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            cvzone.cornerRect(img, (int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)))
            # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (233, 50, 10), 3)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class name
            cls = box.cls[0]
            cvzone.putTextRect(img, f'{classNames[int(cls)]} {conf}', (max(20, int(x1)),max(50, int(y1))), 2, 2)

    # Adding FPS
    cTime = time.time()
    fps = 1 / (cTime - fTime)
    fTime = cTime
    cv2.putText(img, str(int(fps)), (40, 60), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 0, 0), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
