import argparse
import os

import cv2
import numpy as np

from deep_sort import nn_matching, preprocessing, visualization
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from deep_sort.iou_matching import iou_cost

from deep_sort.object_detectors.original_od import OriginalOD
from deep_sort.object_detectors.yolov5_od import YOLOv5OD
from deep_sort.object_detectors.yolov10_od import YOLOv10OD
from deep_sort.object_detectors.nanodet_od import NanodetOD

from deep_sort.feature_generators.original_fg import OriginalFG
from deep_sort.feature_generators.dpreid_fg import DeepPersonReidFG

from deep_sort.types.yolo_model_types import YOLOv5Types, YOLOv10Types
from deep_sort.types.nanodet_types import NanodetModelTypes
from deep_sort.types.dpreid_types import DeepPersonReidTypes

from deep_sort.metrics.fps import FPSMetric
from deep_sort.metrics.classic import ClassicsMetric
from deep_sort.metrics.hota import HotaMetric



from utils.datasets import MOTChallenge


# def gather_sequence_info(sequence_dir, detection_file):
#     """Gather sequence information, such as image filenames, detections,
#     groundtruth (if available).

#     Parameters
#     ----------
#     sequence_dir : str
#         Path to the MOTChallenge sequence directory.
#     detection_file : str
#         Path to the detection file.

#     Returns
#     -------
#     Dict
#         A dictionary of the following sequence information:

#         * sequence_name: Name of the sequence
#         * image_filenames: A dictionary that maps frame indices to image
#           filenames.
#         * detections: A numpy array of detections in MOTChallenge format.
#         * groundtruth: A numpy array of ground truth in MOTChallenge format.
#         * image_size: Image size (height, width).
#         * min_frame_idx: Index of the first frame.
#         * max_frame_idx: Index of the last frame.

#     """
#     image_dir = os.path.join(sequence_dir, "img1")
#     image_filenames = {
#         int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
#         for f in os.listdir(image_dir)}
#     groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

#     detections = None
#     if detection_file is not None:
#         detections = np.load(detection_file)
#     groundtruth = None
#     if os.path.exists(groundtruth_file):
#         groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

#     if len(image_filenames) > 0:
#         image = cv2.imread(next(iter(image_filenames.values())),
#                            cv2.IMREAD_GRAYSCALE)
#         image_size = image.shape
#     else:
#         image_size = None

#     if len(image_filenames) > 0:
#         min_frame_idx = min(image_filenames.keys())
#         max_frame_idx = max(image_filenames.keys())
#     else:
#         min_frame_idx = int(detections[:, 0].min())
#         max_frame_idx = int(detections[:, 0].max())

#     info_filename = os.path.join(sequence_dir, "seqinfo.ini")
#     if os.path.exists(info_filename):
#         with open(info_filename, "r") as f:
#             line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
#             info_dict = dict(
#                 s for s in line_splits if isinstance(s, list) and len(s) == 2)

#         update_ms = 1000 / int(info_dict["frameRate"])
#     else:
#         update_ms = None

#     feature_dim = detections.shape[1] - 10 if detections is not None else 0
#     seq_info = {
#         "sequence_name": os.path.basename(sequence_dir),
#         "image_filenames": image_filenames,
#         "detections": detections,
#         "groundtruth": groundtruth,
#         "image_size": image_size,
#         "min_frame_idx": min_frame_idx,
#         "max_frame_idx": max_frame_idx,
#         "feature_dim": feature_dim,
#         "update_ms": update_ms
#     }
#     return seq_info


# def create_detections(detection_mat, frame_idx, min_height=0):
#     """Create detections for given frame index from the raw detection matrix.

#     Parameters
#     ----------
#     detection_mat : ndarray
#         Matrix of detections. The first 10 columns of the detection matrix are
#         in the standard MOTChallenge detection format. In the remaining columns
#         store the feature vector associated with each detection.
#     frame_idx : int
#         The frame index.
#     min_height : Optional[int]
#         A minimum detection bounding box height. Detections that are smaller
#         than this value are disregarded.

#     Returns
#     -------
#     List[tracker.Detection]
#         Returns detection responses at given frame index.

#     """
#     frame_indices = detection_mat[:, 0].astype(int)
#     mask = frame_indices == frame_idx

#     detection_list = []
#     for row in detection_mat[mask]:
#         bbox, confidence, feature = row[2:6], row[6], row[10:]
#         if bbox[3] < min_height:
#             continue
#         detection_list.append(Detection(bbox, confidence, feature))
#     return detection_list


# def create_detections(detection_mat, frame_idx, min_height=0):
#     frame_indices = detection_mat[:, 0].astype(int)
#     mask = frame_indices == frame_idx

#     detection_list = []
#     for row in detection_mat[mask]:
#         bbox, confidence, feature = row[2:6], row[6], row[10:]
#         if bbox[3] < min_height:
#             continue
#         detection_list.append(Detection(bbox, confidence, feature))
#     return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, detection_model, feature_generator_model, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    dataset = MOTChallenge(sequence_dir)
    seq_info = dataset.get_info()

    match detection_model:
        case "original":
            detector = OriginalOD(sequence_dir)
        case "yolov5nano":
            detector = YOLOv5OD(YOLOv5Types.NANO)
        case "yolov5small":
            detector = YOLOv5OD(YOLOv5Types.SMALL)
        case "yolov5medium":
            detector = YOLOv5OD(YOLOv5Types.MEDIUM)
        case "yolov5large":
            detector = YOLOv5OD(YOLOv5Types.LARGE)
        case "yolov5extralarge":
            detector = YOLOv5OD(YOLOv5Types.EXTRALARGE)
        case "yolov10nano":
            detector = YOLOv10OD(YOLOv10Types.NANO)
        case "yolov10small":
            detector = YOLOv10OD(YOLOv10Types.SMALL)
        case "yolov10medium":
            detector = YOLOv10OD(YOLOv10Types.MEDIUM)
        case "yolov10balanced":
            detector = YOLOv10OD(YOLOv10Types.BALANCED)
        case "yolov10large":
            detector = YOLOv10OD(YOLOv10Types.LARGE)
        case "yolov10extralarge":
            detector = YOLOv10OD(YOLOv10Types.EXTRALARGE)
        case "nanodet":
            detector = NanodetOD(NanodetModelTypes.PLUSM416)
        case _:
            raise ValueError(f"Invalid detection model: {detection_model}")
        
    match feature_generator_model:
        case "original":
            feature_generator = OriginalFG()
        case "dpreid_shufflenet":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.SHUFFLENET)
        case "dpreid_mlfn":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.MLFN)
        case "dpreid_mobilenetv2_1_0":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.MOBILENETv2_x1_0)
        case "dpreid_mobilenetv2_1_4":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.MOBILENETv2_x1_4)
        case "dpreid_osnet_ibn_x1_0":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.OSNET_IBN_x1_0)
        case "dpreid_osnet_ain_x1_0":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.OSNET_AIN_x1_0)
        case "dpreid_osnet_ain_x0_75":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.OSNET_AIN_x0_75)
        case "dpreid_osnet_ain_x0_5":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.OSNET_AIN_x0_5)
        case "dpreid_osnet_ain_x0_25":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.OSNET_AIN_x0_25)
        case "dpreid_osnet_x1_0":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.OSNET_x1_0)
        case "dpreid_osnet_x0_75":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.OSNET_x0_75)
        case "dpreid_osnet_x0_5":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.OSNET_x0_5)
        case "dpreid_osnet_x0_25":
            feature_generator = DeepPersonReidFG(DeepPersonReidTypes.OSNET_x0_25)
        case _:
            raise ValueError(f"Invalid feature generator model: {feature_generator_model}")

    # detector = OriginalOD(sequence_dir)
    # detector = YOLOv5OD(YOLOv5Types.NANO)
    # detector = YOLOv10OD(YOLOv10Types.BALANCED)
    # detector = NanodetOD(NanodetModelTypes.PLUSM416)
    # feature_generator = OriginalFG()
    # feature_generator = DeepPersonReidFG(DeepPersonReidTypes.OSNET_AIN_x0_75)

    # seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    # detections_all = np.load("resources/detections/MOT16_POI_test/MOT16-01.npy")

    metric_detections = []
    metric_gts = seq_info["groundtruth"]

    metric_gts = [[int(i[0])] + i[1:6].tolist() for i in metric_gts if i[6] >= min_confidence]

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = detector.get_detections(seq_info["image_filenames"][frame_idx], min_detection_height)
        # detections = create_detections(
        #     seq_info["detections"], frame_idx, min_detection_height)

        features = feature_generator.get_features(seq_info["image_filenames"][frame_idx], np.array([det.tlwh for det in detections]))
        detections = [Detection(detection.tlwh, detection.confidence, feature) for detection, feature in zip(detections, features)]

        # detections = create_detections(detections_all, frame_idx, 0)

        detections = [d for d in detections if d.confidence >= min_confidence]

        for i in detections:
            metric_detections.append([frame_idx] + i.tlwh.tolist())

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # print(scores)
        # input()

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=seq_info["update_ms"])
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)


    classic_metrics = ClassicsMetric(metric_detections, metric_gts)
    metrics = classic_metrics.get_metric()
    # print(classic_metrics.get_metric())
    hota_metric = HotaMetric(results, metric_gts)
    metrics["hota"] = hota_metric.get_metric()
    # print(hota_metric.get_metric())
    # fps_metric = FPSMetric(visualizer)
    # print(fps_metric.get_fps())

    print(metrics)

    #Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=False)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="tracker.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--detection_model", help="Detection model",
        default="original")
    parser.add_argument(
        "--feature_generator_model", help="Feature generator model",
        default="original")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.detection_model, args.feature_generator_model, args.display)
