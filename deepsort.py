import torch
import cv2
from ultralytics import YOLO

model = YOLO("models/detection/yolov10/yolov10x.pt")

# print(model.info())

img = "MOT16/test/MOT16-01/img1/000001.jpg"

results = model(img)

img_frame = cv2.imread(img)

for result in results:
    detections = []
    for bboxes in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = bboxes

        # detections.append([x1, y1, x2, y2, score])
        if class_id == 0:
            cv2.rectangle(img_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))

cv2.imshow("Image", img_frame)
cv2.waitKey(0)