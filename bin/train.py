""" script to train model """

import cycle_gan
import toml
import os
import argparse

HYPERPARAMETER = os.getenv('HYPERPARAMETER', './bin/hyperparameter.toml')


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-e', '--epoch', help='Epoch.', required=True, type=int, **share_param)
    parser.add_argument('-v', '--version', help='number.', default=None, type=int, **share_param)
    parser.add_argument('--data', help='Dataset.', default='monet2photo', type=str, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    )

    tfrecord_dir = './datasets/tfrecords/%s' % args.data
    ckpt_dir = './checkpoint/%s' % args.data

    if args.version is None:
        hyperparameter = toml.load(open(HYPERPARAMETER))
        checkpoint_dir, _ = cycle_gan.checkpoint_version(ckpt_dir, hyperparameter)
    else:
        checkpoint_dir, hyperparameter = cycle_gan.checkpoint_version(ckpt_dir, version=args.version)

    model_instance = cycle_gan.CycleGAN(tfrecord_dir=tfrecord_dir,
                                        checkpoint_dir=checkpoint_dir,
                                        **hyperparameter)

    model_instance.train(epoch=args.epoch)