import os
import numpy as np
import cv2
from deep_sort.bbox import BBox

class ImageEncoder(object):
    def __init__(
            self, 
            checkpoint_filename, 
            input_name="images",
            output_name="features"):
        self.session = tf.compat.v1.Session()
        with open(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "%s:0" % input_name)
        self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __run_in_batches(self, f, data_dict, out, batch_size):
        data_len = len(out)
        num_batches = int(data_len / batch_size)

        s, e = 0, 0
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
            out[s:e] = f(batch_data_dict)
        if e < len(out):
            batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
            out[e:] = f(batch_data_dict)

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        self.__run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out

class OriginalOD():
    def __init__(self, path):
        self.path = path
        detection_file = os.path.join(path, "det/det.txt")
        self.__detections_in = None
        if os.path.exists(detection_file):
            self.__detections_in = np.loadtxt(detection_file, delimiter=',')
        # self.image_encoder = ImageEncoder("resources/networks/mars-small128.pb")

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
        model_filename, 
        input_name="images",
        output_name="features", 
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
    
    def get_detections(self, image_file, encoder, min_detection_height=0):
        detections_out = []

        # encoder = self.__create_box_encoder("resources/networks/mars-small128.pb", batch_size=32)

        image_index = int(os.path.splitext(os.path.basename(image_file))[0])

        frame_indices = self.__detections_in[:, 0].astype(int)
        mask = frame_indices == image_index

        rows = self.__detections_in[mask]

        # bgr_image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        # features = encoder(bgr_image, rows[:, 2:6].copy())

        # detections_out += [np.r_[(row, feature)] for row, feature in zip(rows, features)]


        # detections_list = []
        # for row in detections_out:
        #     bbox, confidence, feature = row[2:6], row[6], row[10:]
        #     if bbox[3] < min_detection_height:
        #         continue
        #     detections_list.append(Detection(bbox, confidence, feature))

        detections_list = []
        for row in rows:
            bbox, confidence = row[2:6], row[6]
            if bbox[3] < min_detection_height:
                continue
            detections_list.append(BBox(bbox, confidence))

        return detections_list
        

# test = OriginalOD("MOT16/test/MOT16-02/")
# print(test.get_detections("000001.jpg"))
