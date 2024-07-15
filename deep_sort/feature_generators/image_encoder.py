import numpy as np
import tensorflow as tf

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