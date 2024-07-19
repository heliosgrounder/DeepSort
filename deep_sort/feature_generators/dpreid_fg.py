import numpy as np
import cv2

from torchreid.utils import FeatureExtractor

from deep_sort.types.dpreid_types import DeepPersonReidTypes

class DeepPersonReidFG():
    def __init__(self, model_type):
        match model_type:
            case DeepPersonReidTypes.SHUFFLENET:
                model_name = "shufflenet"
                model_path = "models/feature_generation/deeppersonreid/shufflenet.pth.tar"
            case DeepPersonReidTypes.MOBILENETv2_x1_0:
                model_name = "mobilenetv2_x1_0"
                model_path = "models/feature_generation/deeppersonreid/mobilenetv2_1.0.pth"
            case DeepPersonReidTypes.MOBILENETv2_x1_4:
                model_name = "mobilenetv2_x1_4"
                model_path = "models/feature_generation/deeppersonreid/mobilenetv2_1.4.pth"
            case DeepPersonReidTypes.MLFN:
                model_name = "mlfn"
                model_path = "models/feature_generation/deeppersonreid/mlfn.pth.tar"
            case DeepPersonReidTypes.OSNET_x1_0:
                model_name = "osnet_x1_0"
                model_path = "models/feature_generation/deeppersonreid/osnet_x1_0_imagenet.pth"
            case DeepPersonReidTypes.OSNET_x0_75:
                model_name = "osnet_x0_75"
                model_path = "models/feature_generation/deeppersonreid/osnet_x0_75_imagenet.pth"
            case DeepPersonReidTypes.OSNET_x0_5:
                model_name = "osnet_x0_5"
                model_path = "models/feature_generation/deeppersonreid/osnet_x0_5_imagenet.pth"
            case DeepPersonReidTypes.OSNET_x0_25:
                model_name = "osnet_x0_25"
                model_path = "models/feature_generation/deeppersonreid/osnet_x0_25_imagenet.pth"
            case DeepPersonReidTypes.OSNET_IBN_x1_0:
                model_name = "osnet_ibn_x1_0"
                model_path = "models/feature_generation/deeppersonreid/osnet_ibn_x1_0_imagenet.pth"
            case DeepPersonReidTypes.OSNET_AIN_x1_0:
                model_name = "osnet_ain_x1_0"
                model_path = "models/feature_generation/deeppersonreid/osnet_ain_x1_0_imagenet.pth"
            case DeepPersonReidTypes.OSNET_AIN_x0_75:
                model_name = "osnet_ain_x0_75"
                model_path = "models/feature_generation/deeppersonreid/osnet_ain_x0_75_imagenet.pyth"
            case DeepPersonReidTypes.OSNET_AIN_x0_5:
                model_name = "osnet_ain_x0_5"
                model_path = "models/feature_generation/deeppersonreid/osnet_ain_x0_5_imagenet.pyth"
            case DeepPersonReidTypes.OSNET_AIN_x0_25:
                model_name = "osnet_ain_x0_25"
                model_path = "models/feature_generation/deeppersonreid/osnet_ain_x0_25_imagenet.pyth"
            case _:
                model_name = None
                model_path = None


        self.__extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device="cuda"
        )
    
    def __extract_image_patch(self, image, bbox, patch_shape):
        """Extract image patch from bounding box.

        Parameters
        ----------
        image : ndarray
            The full image.
        bbox : array_like
            The bounding box in format (x, y, width, height).
        patch_shape : Optional[array_like]
            This parameter can be used to enforce a desired patch shape
            (height, width). First, the `bbox` is adapted to the aspect ratio
            of the patch shape, then it is clipped at the image boundaries.
            If None, the shape is computed from :arg:`bbox`.

        Returns
        -------
        ndarray | NoneType
            An image patch showing the :arg:`bbox`, optionally reshaped to
            :arg:`patch_shape`.
            Returns None if the bounding box is empty or fully outside of the image
            boundaries.

        """
        bbox = np.array(bbox)
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image

    def get_features(self, image_file, boxes):
        bgr_image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image_patches = []
        for box in boxes:
            patch = self.__extract_image_patch(bgr_image, box, bgr_image.shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        # image_patches = np.asarray(image_patches)
        features = self.__extractor(image_patches)
        return features.cpu().numpy()
