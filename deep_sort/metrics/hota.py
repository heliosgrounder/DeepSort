import numpy as np
from scipy.optimize import linear_sum_assignment

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

class HotaMetric():
    def __init__(self, detections, gts):
        self.__detections = detections
        self.__gts = gts
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

        self.__frame_dict = dict(sorted(self.__frame_dict.items()))

    def get_metric(self):
        return self.__calculate_metric()

    def __calculate_metric(self):
        result = np.zeros(shape=(19, 4))

        for frame_id in self.__frame_dict.keys():
            result += self.__calculate_frame(frame_id)

        detection = result[:, 0] / np.maximum(1, np.sum(result[:, 0:3], axis=1))
        association = np.divide(result[:, 3], result[:, 0], out=np.zeros_like(result[:, 0]), where=result[:, 0] > 0)

        return 1 / 19 * np.sum(np.sqrt(detection * association))

    def __calculate_frame(self, frame_id):
        detection_per_frame = self.__frame_dict[frame_id]["det"]
        gt_per_frame = self.__frame_dict[frame_id]["gt"]

        if len(detection_per_frame) == 0 and len(gt_per_frame) == 0:
            return np.repeat([[0, 0, 0, 1]], repeats=19, axis=0)
        elif len(detection_per_frame) == 0:
            return np.repeat([[0, 0, len(gt_per_frame), 0]], repeats=19, axis=0)
        elif len(gt_per_frame) == 0:
            return np.repeat([[0, len(detection_per_frame), 0, 0]], repeats=19, axis=0)

        matrix = np.zeros((len(detection_per_frame), len(gt_per_frame)), dtype=np.float32)

        for i in range(len(detection_per_frame)):
            for j in range(len(gt_per_frame)):
                matrix[i, j] = iou(detection_per_frame[i][1:], gt_per_frame[j][1:])

        row_idx, col_idx = linear_sum_assignment(matrix, maximize=True)

        partial_result = np.zeros(shape=(19, 4))

        for i, a in enumerate(np.arange(0.05, 1.00, 0.05)):
            for r, c in zip(row_idx, col_idx):
                if matrix[r, c] >= a:
                    partial_result[i, 0] += 1
                    partial_result[i, 3] += self.__calculate_track(a, detection_per_frame[r][1], gt_per_frame[c][1])

            partial_result[i, 1] = len(detection_per_frame) - partial_result[i, 0]
            partial_result[i, 2] = len(gt_per_frame) - partial_result[i, 0]

        return partial_result

    def __calculate_track(self, a, det_tid, gt_tid):
        tpa = 0
        fpa = 0
        fna = 0
        di = 0
        gi = 0

        detection_per_frame = [i for i in self.__detections if i[1] == det_tid]
        gt_per_frame = [i for i in self.__gts if i[1] == gt_tid]

        if len(detection_per_frame) == 0 and len(gt_per_frame) == 0:
            return 1
        
        while di < len(detection_per_frame) and gi < len(gt_per_frame):
            det = detection_per_frame[di]
            gt = gt_per_frame[gi]

            if det[0] == gt[0]:
                if a <= iou(det[2:], gt[2:]):
                    tpa += 1
                else:
                    fpa += 1
                    fna += 1

                di += 1
                gi += 1
            elif det[0] < gt[0]:
                fpa += 1
                di += 1
            else:
                fna += 1
                gi += 1

        while di < len(detection_per_frame):
            fpa += 1
            di += 1

        while gi < len(gt_per_frame):
            fna += 1
            gi += 1

        return tpa / (tpa + fna + fpa)