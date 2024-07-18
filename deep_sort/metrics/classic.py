import numpy as np
from scipy.optimize import linear_sum_assignment

# from deep_sort.iou_matching import iou

def iou(bbox_1, bbox_2):
    b_1 = [0, 0, 0, 0]
    b_1[0] = bbox_1[0]
    b_1[1] = bbox_1[1]
    b_1[2] = bbox_1[0] + bbox_1[2]
    b_1[3] = bbox_1[1] + bbox_1[3]

    b_2 = [0, 0, 0, 0]
    b_2[0] = bbox_2[0]
    b_2[1] = bbox_2[1]
    b_2[2] = bbox_2[0] + bbox_2[2]
    b_2[3] = bbox_2[1] + bbox_2[3]

    x_a = max(b_1[0], b_2[0])
    y_a = max(b_1[1], b_2[1])
    x_b = max(b_1[2], b_2[2])
    y_b = max(b_1[3], b_2[3])

    intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (b_1[2] - b_1[0] + 1) * (b_1[3] - b_1[1] + 1)
    box_b_area = (b_2[2] - b_2[0] + 1) * (b_2[3] - b_2[1] + 1)

    iou = intersection_area / float(box_a_area + box_b_area - intersection_area)

    return iou

class ClassicsMetric():
    def __init__(self, detections, gts, min_iou=0.5):
        # self.__detections = detections
        # self.__gts = gts
        self.__min_iou = min_iou

        self.__frame_dict = dict()
        for det in detections:
            if det[0] not in self.__frame_dict.keys():
                self.__frame_dict[det[0]] = {
                    "det": [],
                    "gt": []
                }
            self.__frame_dict[det[0]]["det"].append(det[1:])
        for gt in gts:
            if gt[0] not in self.__frame_dict.keys():
                self.__frame_dict[gt[0]] = {
                    "det": [],
                    "gt": []
                }
            self.__frame_dict[gt[0]]["gt"].append(gt[1:])

        mega_confusion = [0, 0, 0]
        for frame_id in self.__frame_dict.keys():
            confusion = self.__calculate_metric(frame_id)
            mega_confusion[0] += confusion[0]
            mega_confusion[1] += confusion[1]
            mega_confusion[2] += confusion[2]
        
        self.__precision = 0
        self.__recall = 0
        self.__f1_score = 0

        if mega_confusion[0] == 0 and mega_confusion[1] == 0:
            self.__precision = 1
        else:
            self.__precision = 1.0 * mega_confusion[0] / (mega_confusion[0] + mega_confusion[1])
        if mega_confusion[0] == 0 and mega_confusion[2] == 0:
            self.__recall = 1
        else:
            self.__recall = 1.0 * mega_confusion[0] / (mega_confusion[0] + mega_confusion[2])

        if self.__precision == 0 and self.__recall == 0:
            self.__f1_score = 1
        else:
            self.__f1_score = 2.0 * self.__precision * self.__recall / (self.__precision + self.__recall)


    def get_metric(self):
        return {"precision": self.__precision, "recall": self.__recall, "f1-score": self.__f1_score}

    def __calculate_metric(self, frame_id):
        detection_per_frame = self.__frame_dict[frame_id]["det"]
        gt_per_frame = self.__frame_dict[frame_id]["gt"]

        confusion = [0, 0, 0]
        if len(detection_per_frame) == 0 and len(gt_per_frame) == 0:
            return confusion
        elif len(detection_per_frame) == 0:
            confusion[2] = len(gt_per_frame)
            return confusion
        elif len(gt_per_frame) == 0:
            confusion[1] = len(detection_per_frame)
            return confusion

        matrix = np.zeros((len(detection_per_frame), len(gt_per_frame)), dtype=np.float32)

        for i in range(len(detection_per_frame)):
            for j in range(len(gt_per_frame)):
                matrix[i, j] = iou(detection_per_frame[i], gt_per_frame[j])

        row_idx, col_idx = linear_sum_assignment(matrix, maximize=True)

        for r, c in zip(row_idx, col_idx):
            if self.__min_iou <= matrix[r, c]:
                confusion[0] += 1
        
        confusion[1] = len(detection_per_frame) - confusion[0]
        confusion[2] = len(gt_per_frame) - confusion[0]

        return confusion






class PrecisionMetric():
    def __init__(self):
        pass

    def get_metric(self):
        pass


class RecallMetric():
    def __init__(self):
        pass
    
    def get_metric(self):
        pass
    

class F1Metric():
    def __init__(self):
        pass
    
    def get_metric(self):
        pass
