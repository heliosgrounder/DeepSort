from deep_sort.bbox import BBox
from ultralytics import YOLO
from deep_sort.types.yolo_model_types import YOLOv10Types

class YOLOv10OD():
    def __init__(self, model_type):
        match model_type:
            case YOLOv10Types.NANO:
                path = "models/detection/yolov10/yolov10n.pt"
            case YOLOv10Types.SMALL:
                path = "models/detection/yolov10/yolov10s.pt"
            case YOLOv10Types.MEDIUM:
                path = "models/detection/yolov10/yolov10m.pt"
            case YOLOv10Types.BALANCED:
                path = "models/detection/yolov10/yolov10b.pt"
            case YOLOv10Types.LARGE:
                path = "models/detection/yolov10/yolov10l.pt"
            case YOLOv10Types.EXTRALARGE:
                path = "models/detection/yolov5/yolov10x.pt"
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