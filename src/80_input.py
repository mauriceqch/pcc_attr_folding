import utils
import sklearn
import tensorflow.compat.v1 as tf
import numpy as np


def tf_dataset(batch_pc_gen):
    while True:
        yield next(batch_pc_gen)


def get_dataset(batch_pc_gen, batch_size):
    with tf.device('/device:CPU:0'):
        ds = tf.data.Dataset.from_generator(lambda: tf_dataset(batch_pc_gen), tf.float32, (batch_size, None, 6))
        if tf.test.is_gpu_available():
            ds = ds.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0', 4))

    return ds


def pc_batcher(x):
    return np.array(list(y[0][:, :6] for y in x))


class Dataset:
    def __init__(self, gen, batch_size):
        # Set up data pipelines
        self.gen = gen
        self.batcher = lambda x: pc_batcher(x)
        self.batch_gen = utils.generators.BatchGenerator(self.gen, batch_size, self.batcher)

        # TF graph data pipeline
        self.batch_ds = get_dataset(self.batch_gen, batch_size)
        self.iterator = tf.data.make_one_shot_iterator(self.batch_ds)

        self.output_types = tf.data.get_output_types(self.batch_ds)
        self.output_shapes = tf.data.get_output_shapes(self.batch_ds)
        self.string_handle = self.iterator.string_handle


class InputPipeline:
    def __init__(self, files, batch_size, train_test_split, infinite_data, test_size=0.1):
        self.files = files

        def data_transform(data):
            if infinite_data:
                return utils.generators.sampling_generator(data)
            else:
                return iter(data)

        if train_test_split:
            files_train, files_test = sklearn.model_selection.train_test_split(files, test_size=test_size)
            self.len_train, self.len_test = len(files_train) // batch_size, len(files_test) // batch_size
            data_train = utils.pc_io.load_points(files_train)
            data_test = utils.pc_io.load_points(files_test)
            self.pc_ds_test = Dataset(data_transform(data_test), batch_size)
            self.pc_ds_train = Dataset(data_transform(data_train), batch_size)

            # Train/Test switching
            self.handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(self.handle, self.pc_ds_train.output_types,
                                                           self.pc_ds_train.output_shapes)
            self.next_element = iterator.get_next()
        else:
            self.data = utils.pc_io.load_points(files)
            self.pc_ds = Dataset(data_transform(self.data), batch_size)
            self.next_element = self.pc_ds.iterator.get_next()
