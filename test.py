from deep_sort.object_detectors.yolov5_od import YOLOv5OD
from deep_sort.types.yolo_model_types import YOLOv5Types

test = YOLOv5OD(YOLOv5Types.NANO)

print(test.get_detections("MOT16/test/MOT16-01/img1/000001.jpg"))
