import numpy as np
import cv2

from deep_sort.feature_generators.image_encoder import ImageEncoder

class OriginalFG():
    def __init__(self, model_filename="resources/networks/mars-small128.pb", batch_size=32):
        self.image_encoder = ImageEncoder(model_filename)
        self.encoder = self.__create_box_encoder(batch_size)

    def get_features(self, image_file, boxes):
        bgr_image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        return self.encoder(bgr_image, boxes)

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

    def __create_box_encoder(
        self, 
        batch_size=32):
        # image_encoder = ImageEncoder(model_filename, input_name, output_name)
        image_shape = self.image_encoder.image_shape

        def encoder(image, boxes):
            image_patches = []
            for box in boxes:
                patch = self.__extract_image_patch(image, box, image_shape[:2])
                if patch is None:
                    print("WARNING: Failed to extract image patch: %s." % str(box))
                    patch = np.random.uniform(
                        0., 255., image_shape).astype(np.uint8)
                image_patches.append(patch)
            image_patches = np.asarray(image_patches)
            return self.image_encoder(image_patches, batch_size)

        return encoder