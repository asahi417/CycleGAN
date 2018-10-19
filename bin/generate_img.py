""" script to generate image from trained model """

import cycle_gan
import os
import argparse
# from PIL import Image
import numpy as np
import scipy.misc as misc


HYPERPARAMETER = os.getenv('HYPERPARAMETER', './bin/hyperparameter.toml')


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-v', '--version', help='number.', default=0, type=int, **share_param)
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

    checkpoint_dir, hyperparameter = cycle_gan.checkpoint_version(ckpt_dir, version=args.version)
    model_instance = cycle_gan.CycleGAN(tfrecord_dir=tfrecord_dir,
                                        checkpoint_dir=checkpoint_dir,
                                        **hyperparameter)
    images = model_instance.generate_img(10)

    batch = 4
    n = 4  # cols
    m = int(2 * batch)  # rows
    img_size = (256, 256, 3)

    canvas = 255 * np.ones((m * img_size[0] + (10 * m) + 10, n * img_size[1] + (10 * n) + 10, 3),
                           dtype=np.uint8)

    start_x = 10
    start_y = 10

    x = 0
    y = 0
    i = 0

    images = model_instance.generate_img(batch)
    for image in images:
        for img in image:
            if i + 1 > n*m:
                break

            end_x = start_x + 256
            end_y = start_y + 256
            canvas[start_y:end_y, start_x:end_x, :] = img

            if x < n:
                start_x += 256 + 10
                x += 1
            if x == n:
                x = 0
                start_x = 10
                start_y = end_y + 10
                end_y = start_y + 256
            i += 1

    misc.imsave('./bin/img/generated_img/%s-v%s.jpg' % (args.data, args.version), canvas)

    # Image.fromarray(img, 'RGB').save('./bin/img/generated_img/%s-%s-%i.png' % (args.model, args.data, n))