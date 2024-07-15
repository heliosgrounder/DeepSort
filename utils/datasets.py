import os
import numpy as np
import cv2

class MOTChallenge():
    """
    This class represents a MOT Chalange structure handler.

    Parameters
    ----------

    path : str
        Path to the MOTChallenge sequence directory.

    """
    def __init__(self, path):
        self.path = path

    def get_info(self):
        image_dir = os.path.join(self.path, "img1")
        image_filenames = { 
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f) for f in os.listdir(image_dir)
        }
        
        # detections = None
        # if detection_file is not None:
        #     detections = np.load(detection_file)

        groundtruth_file = os.path.join(self.path, "gt/gt.txt")
        groundtruth = None
        if os.path.exists(groundtruth_file):
            groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

        info_filename = os.path.join(self.path, "seqinfo.ini")
        if os.path.exists(info_filename):
            with open(info_filename, "r") as f:
                line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
                info_dict = dict([s for s in line_splits if isinstance(s, list) and len(s) == 2])

            sequence_name = info_dict["name"]
            image_size = (info_dict["imHeight"], info_dict["imWidth"])
            update_ms = 1000 / int(info_dict["frameRate"])
            min_frame_idx = 1
            max_frame_idx = int(info_dict["seqLength"])
            image_ext = info_dict["imExt"]
        else:
            sequence_name = os.path.basename(self.path)
            if len(image_filenames):
                image = cv2.imread(next(iter(image_filenames.values())), cv2.IMREAD_GRAYSCALE)
                image_size = image.shape
            else:
                image_size = None
            min_frame_idx = -1
            max_frame_idx = -1
            if len(image_filenames):
                min_frame_idx = min(image_filenames.keys())
                max_frame_idx = max(image_filenames.keys())
            update_ms = None
            image_ext = None

        # feature_dim = detections.shape[1] - 10 if detections is not None else 0
        info = {
            "sequence_name": sequence_name,
            "image_filenames": image_filenames,
            # "detections": detections,
            "groundtruth": groundtruth,
            "image_size": image_size,
            "min_frame_idx": min_frame_idx,
            "max_frame_idx": max_frame_idx,
            # "feature_dim": feature_dim,
            "update_ms": update_ms,
            "image_ext": image_ext
        }
        return info
