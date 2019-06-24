# -*- coding: utf-8 -*-

import argparse
import numpy as np

from sgbm import SGBMatcher
from reconstruct import Reconstructor, imread


# python test.py img/rgb_2011_09_26_0001_02_0070.png img/rgb_2011_09_26_0001_02_0070.png

def parse_args():
    parser = argparse.ArgumentParser(description='Semi-Global Matching')
    parser.add_argument('left_image_name')
    parser.add_argument('right_image_name')
    parser.add_argument('-M', default=1, type=int, help='The sparsity of y-axis of image')
    parser.add_argument('-N', default=1, type=int, help='The sparsity of x-axis of image')

    parser.add_argument('-f', '--focal_length', default=500, type=float)
    parser.add_argument('-d', '--distance_between_cameras', default=10, type=float)

    parser.add_argument('--num_disparities', default=80, type=int,
            help='The max_disp in use, must be dividable by 16. default: 16.')
    parser.add_argument('--block_size', default=5, type=int,
            help='Block size in calculation. default: 5.')
    parser.add_argument('--window_size', default=15, choices=[3, 5, 7, 15],
            type=int,
            help='How large a search window is. default: 5.')

    parser.add_argument('--show_disp_map', action='store_true', default=False, 
            help='Visualize disparity results')
    parser.add_argument('--show_depth_map', action='store_true', default=False, 
            help='Visualize depth results')
    args = parser.parse_args()

    assert args.num_disparities % 16 == 0, "The max_disp must be dividable by 16."

    return args


def main():
    args = parse_args()

    img_l = imread(args.left_image_name, 1)
    img_r = imread(args.right_image_name, 1)

    # Init SGBM
    sgm = SGBMatcher(
            num_disparities=args.num_disparities,
            block_size=args.block_size,
            window_size=args.window_size)

    sgm.show_param()

    # Get disparity
    disp = sgm.compute(img_l, img_r)

    # Sparsify the computated points
    disp_ = np.zeros_like(disp)
    disp_[::args.M, ::args.N] = disp[::args.M,::args.N]

    # Init Visualizer
    rec = Reconstructor(
            img_l=img_l, 
            img_r=img_r, 
            disp=disp, 
            focal_length=args.focal_length, 
            baseline=args.distance_between_cameras,
            show_disp_map=args.show_disp_map,
            show_depth_map=args.show_depth_map
            )

    rec.visualize()


if __name__ == '__main__':
    main()
