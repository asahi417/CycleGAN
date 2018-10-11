"""
check dataset image
"""

import os
import argparse
import numpy as np
from PIL import Image
from glob import glob

OUTPUT = os.getenv('OUTPUT', './bin/img/check_image')

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT, exist_ok=True)


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-n', '--num', help='number.', default=10, type=int, **share_param)
    parser.add_argument('-c', '--crop', help='number.', default=None, type=int, **share_param)
    parser.add_argument('-r', '--resize', help='number.', default=None, type=int, **share_param)
    parser.add_argument('--data', help='Dataset.', required=True, type=str, **share_param)
    parser.add_argument('--type', help='Type [trainA, trainB,...].', required=True, type=str, **share_param)
    return parser.parse_args()


def load_and_save(n,
                  data_name,
                  data_type,
                  crop,
                  resize):

    image_files = glob('./datasets/%s/%s/*.jpg' % (data_name, data_type))
    image_files += glob('./datasets/%s/%s/*.png' % (data_name, data_type))
    image_filenames = sorted(image_files)
    print('total_size:', len(image_filenames))

    for number, image_path in enumerate(image_filenames):
        # open as pillow instance
        image = Image.open(image_path)
        w, h = image.size

        if crop is not None:
            # cropping
            upper = int(np.floor(h / 2 - crop / 2))
            lower = int(np.floor(h / 2 + crop / 2))
            left = int(np.floor(w / 2 - crop / 2))
            right = int(np.floor(w / 2 + crop / 2))
            image = image.crop((left, upper, right, lower))

        if resize is not None:
            # resize
            image = image.resize((resize, resize))

        # pillow instance -> numpy array
        image = np.array(image)
        print(image.shape)

        # numpy array -> pillow instance
        image = Image.fromarray(image.astype('uint8'), 'RGB')

        # save it
        name = image_path.split('/')[-1]
        image.save('%s/%s-%s-%s' % (OUTPUT, data_name, data_type, name))
        if number == n:
            break


if __name__ == '__main__':
    args = get_options(
        argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter))
    load_and_save(args.num,
                  data_name=args.data,
                  data_type=args.type,
                  crop=args.crop,
                  resize=args.resize)
