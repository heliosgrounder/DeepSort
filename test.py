import numpy as np

from deep_sort.object_detectors.yolov5_od import YOLOv5OD
from deep_sort.types.yolo_model_types import YOLOv5Types

test_c = YOLOv5OD(YOLOv5Types.NANO)

detections = test_c.get_detections("MOT16/test/MOT16-01/img1/000001.jpg")

# print(test.get_detections("MOT16/test/MOT16-01/img1/000001.jpg"))

# from deep_sort.object_detectors.nanodet_od import NanodetOD
# from deep_sort.types.nanodet_types import NanodetModelTypes

# test = NanodetOD(NanodetModelTypes.PLUSM416)

# print()
# print(test.get_detections("MOT16/test/MOT16-01/img1/000001.jpg"))


from deep_sort.feature_generators.dpreid_fg import DeepPersonReidFG

test = DeepPersonReidFG(-1)

features = test.get_features("MOT16/test/MOT16-01/img1/000001.jpg", np.array([det.tlwh for det in detections]))

print(features.numpy())