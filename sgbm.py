# -*- coding: utf-8 -*-

import argparse
import cv2
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Semi-Global Matching')
    parser.add_argument('left_image_name')
    parser.add_argument('right_image_name')
    parser.add_argument('--num_disparities', default=80, type=int,
            help='The max_disp in use, must be dividable by 16. default: 16.')
    parser.add_argument('--block_size', default=5, type=int,
            help='Block size in calculation. default: 5.')
    parser.add_argument('--window_size', default=15, choices=[3, 5, 7, 15],
            type=int,
            help='How large a search window is. default: 5.')
    args = parser.parse_args()

    assert args.num_disparities % 16 == 0, "The max_disp must be dividable by 16."

    return args


class SGBMatcher():

    def __init__(self,
            min_disparity=0,
            num_disparities=16,
            block_size=5,
            window_size=3,
            disp12_max_diff=1,
            uniqueness_ratio=15,
            speckle_window_size=0,
            speckle_range=2,
            pre_filter_cap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY):

        # SGBM Parameters
        # http://answers.opencv.org/question/182049/pythonstereo-disparity-quality-problems/?answer=183650#post-id-183650
        # window_size: default 3; 5 Works nicely
        #              7 for SGBM reduced size image; 
        #              15 for SGBM full size image (1300px and above)
        # num_disparity: max_disp has to be dividable by 16 f. E. HH 192, 256

        p1 = 8 * 3 * window_size ** 2
        p2 = 32 * 3 * window_size ** 2
        self.param = {
            'minDisparity' : min_disparity,
            'numDisparities' : num_disparities,
            'blockSize' : block_size,
            'windowSize' : window_size,
            'P1' : p1,
            'P2' : p2,
            'disp12MaxDiff' : disp12_max_diff,
            'uniquenessRatio' : uniqueness_ratio,
            'speckleWindowSize' : speckle_window_size,
            'speckleRange' : speckle_range,
            'preFilterCap' : pre_filter_cap,
            'mode' : mode
            }

        param = self.param.copy()
        del param['windowSize']

        self.left_matcher = cv2.StereoSGBM_create(**param)
    
    def show_param(self):
        print(json.dumps(self.param, indent=4, sort_keys=True))

    def visualize(self, img_l, img_r):
        disp_l = self.compute(img_l, img_r)
        disp_l_img = self.normalize(disp_l)
        self.display(img_l, img_r, disp_l_img)

    def compute(self, img_l, img_r):
        return self.left_matcher.compute(img_l, img_r).astype(np.float32) / 16.0

    def normalize(self, disp):
        disp_img = disp.copy()
        cv2.normalize(
                src=disp_img, 
                dst=disp_img, 
                beta=0, 
                alpha=255, 
                norm_type=cv2.NORM_MINMAX
                )
        return np.uint8(disp_img)

    def display(self, img_l, img_r, disp):
        key = 0
        while key not in [27, 32]:
            cv2.imshow("imgl", img_l)
            cv2.imshow("imgr", img_r)
            cv2.imshow("disp", disp)
            key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
        return key


def imread(img_name):
    img =  cv2.imread(img_name, -1)
    assert img is not None, "{} is not found!".format(img_name)

    return img


def main():
    args = parse_args()

    sgm = SGBMatcher(
            num_disparities=args.num_disparities,
            block_size=args.block_size,
            window_size=args.window_size)

    sgm.show_param()

    img_l = imread(args.left_image_name)
    img_r = imread(args.right_image_name)

    sgm.visualize(img_l, img_r)


if __name__ == '__main__':
    main()
