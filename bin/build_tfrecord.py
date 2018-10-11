""" script to build tfrecord """

import cycle_gan
import os
import argparse

PATH_TFRECORD = os.getenv('PATH_TFRECORD', './datasets/tfrecords')


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))
    recorder = cycle_gan.TFRecorder(dataset_name=args.data,
                                    path_to_dataset='./datasets/%s' % args.data,
                                    tfrecord_dir=PATH_TFRECORD)
    recorder.create()
