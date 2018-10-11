import os
from glob import glob
from time import time
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from .util import mkdir


DATA_TYPE = ['trainB', 'trainA', 'testA', 'testB']


def shuffle_data(data, seed=0):
    np.random.seed(seed)
    np.random.shuffle(data)
    return data


def tfrecord_parser(img_shape):
    def __tfrecord_parser(example_proto):
        features = dict(image=tf.FixedLenFeature([], tf.string, default_value=""))
        parsed_features = tf.parse_single_example(example_proto, features)
        feature_image = tf.decode_raw(parsed_features["image"], tf.uint8)
        feature_image = tf.cast(feature_image, tf.float32)
        image = tf.reshape(feature_image, img_shape)
        return image
    return __tfrecord_parser


class TFRecorder:
    """ Formatting data as TFrecord """

    def __init__(self,
                 dataset_name: str,
                 path_to_dataset: str,
                 tfrecord_dir: str,
                 print_progress: bool = True,
                 progress_interval: int = 10):

        self.dataset_name = dataset_name
        self.path_to_dataset = path_to_dataset

        self.path_to_save = '%s/%s' % (tfrecord_dir, dataset_name)
        mkdir(self.path_to_save)

        self.print_progress = print_progress
        self.progress_interval = progress_interval

    def my_print(self, *args, **kwargs):
        if self.print_progress:
            print(*args, **kwargs)

    def create(self):

        def write(image_filenames, name):
            full_size = len(image_filenames)
            self.my_print('writing %s as tfrecord: size %i' % (self.dataset_name, full_size))
            compress_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
            with tf.python_io.TFRecordWriter(name, options=compress_opt) as writer:
                time_stamp = time()
                time_stamp_start = time()
                for n, single_image_path in enumerate(image_filenames):
                    # open as pillow instance
                    image = Image.open(single_image_path)
                    img = np.asarray(image)
                    if img.shape != (256, 256, 3):
                        if img.shape == (256, 256):
                            img = np.tile(np.expand_dims(img, 2), [1, 1, 3])
                            self.my_print('converted gray scale:', img.shape)
                        else:
                            raise ValueError('Error: inconsistency shape', single_image_path, img.shape)

                    img = np.rint(img).clip(0, 255).astype(np.uint8)

                    if n % self.progress_interval == 0:
                        progress_perc = n / full_size * 100
                        cl_time = time() - time_stamp
                        whole_time = time() - time_stamp_start
                        time_per_sam = cl_time / self.progress_interval
                        self.my_print('%d / %d (%0.1f %%), %0.4f sec/image (%0.1f sec) \r'
                                      % (n, full_size, progress_perc, time_per_sam, whole_time),
                                      end='', flush=True)
                        time_stamp = time()

                    ex = tf.train.Example(
                        features=tf.train.Features(
                            feature=dict(image=tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])))
                        )
                    )
                    writer.write(ex.SerializeToString())
            w, h = image.size
            return w, h

        __shape = dict()
        __size = dict()
        for __type in DATA_TYPE:
            self.my_print(__type)
            image_files = sorted(glob('%s/%s/*.jpg' % (self.path_to_dataset, __type)))
            width, height = write(image_files, '%s/%s.tfrecord' % (self.path_to_save, __type))
            __shape[__type] = [width, height]
            __size[__type] = len(image_files)

        meta = dict()
        meta['shape'] = __shape
        meta['size'] = __size

        with open('%s/meta.json' % self.path_to_save, 'w') as outfile:
            json.dump(meta, outfile)
