from deep_sort.bbox import BBox
from ultralytics import YOLO
from deep_sort.types.yolo_model_types import YOLOv5Types

class YOLOv5OD():
    def __init__(self, model_type):
        match model_type:
            case YOLOv5Types.NANO:
                path = "models/detection/yolov5/yolov5nu.pt"
            case YOLOv5Types.SMALL:
                path = "models/detection/yolov5/yolov5su.pt"
            case YOLOv5Types.MEDIUM:
                path = "models/detection/yolov5/yolov5mu.pt"
            case YOLOv5Types.LARGE:
                path = "models/detection/yolov5/yolov5lu.pt"
            case YOLOv5Types.EXTRALARGE:
                path = "models/detection/yolov5/yolov5xu.pt"
            case _:
                path = None

        self.model = YOLO(path)

    def get_detections(self, image_file, min_detection_height=0):
        detections_list = []

        results = self.model(image_file)
        for result in results:
            for bboxes in result.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = bboxes
                bbox = [x1, y1, x2 - x1, y2 - y1]
                if class_id != 0 or bbox[3] < min_detection_height:
                    continue
                detections_list.append(BBox(bbox, confidence))

        return detections_list

                    