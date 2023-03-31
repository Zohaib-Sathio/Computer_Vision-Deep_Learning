from ultralytics import YOLO
import cv2
import cvzone
import time
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO('../YOLO_Weights/yolov8n.pt')

fTime = 0
cTime = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (233, 50, 10), 3)

            cvzone.putTextRect(img, str(math.ceil(box.conf[0]*100)/100),(100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255))

    cTime = time.time()
    fps = 1 / (cTime - fTime)
    fTime = cTime
    cv2.putText(img, str(int(fps)), (40, 60), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
