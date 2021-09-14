# -*- coding: utf-8 -*-

import argparse

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.transform import Rotation as R

from sgbm import SGBMatcher


def parse_args():
    parser = argparse.ArgumentParser(description='Semi-Global Matching')
    parser.add_argument('left_image_name')
    parser.add_argument('right_image_name')
    parser.add_argument('-M',
                        default=1,
                        type=int,
                        help='The sparsity of y-axis of image')
    parser.add_argument('-N',
                        default=1,
                        type=int,
                        help='The sparsity of x-axis of image')

    parser.add_argument('-f', '--focal_length', default=500, type=float)
    parser.add_argument('-d',
                        '--distance_between_cameras',
                        default=10,
                        type=float)

    parser.add_argument(
        '--num_disparities',
        default=80,
        type=int,
        help='The max_disp in use, must be dividable by 16. default: 16.')
    parser.add_argument('--block_size',
                        default=5,
                        type=int,
                        help='Block size in calculation. default: 5.')
    parser.add_argument('--window_size',
                        default=15,
                        choices=[3, 5, 7, 15],
                        type=int,
                        help='How large a search window is. default: 5.')

    parser.add_argument('--show_disp_map',
                        action='store_true',
                        default=False,
                        help='Visualize disparity results')
    parser.add_argument('--show_depth_map',
                        action='store_true',
                        default=False,
                        help='Visualize depth results')
    args = parser.parse_args()

    assert args.num_disparities % 16 == 0, "The max_disp must be dividable by 16."

    return args


class Reconstructor():
    def __init__(self,
                 img_l,
                 img_r,
                 disp,
                 focal_length,
                 baseline,
                 show_disp_map=False,
                 show_depth_map=False,
                 trans_speed=50,
                 angle_speed=np.pi / 180):
        self.img_l = img_l
        self.img_r = img_r
        self.disp = disp
        self.focal_length = focal_length
        self.baseline = baseline
        self.show_disp_map = show_disp_map
        self.show_depth_map = show_depth_map

        self.height, self.width, _ = img_l.shape
        self.q = np.array([[1, 0, 0, -self.width / 2],
                           [0, 1, 0, -self.height / 2],
                           [0, 0, 0, focal_length], [0, 0, -1 / baseline, 0]])
        self.k = np.array([[focal_length, 0, self.width / 2],
                           [0, focal_length, self.height / 2], [0, 0, 1]])
        self.dist_coeff = np.zeros((4, 1))

        self.r = np.eye(3)
        self.t = np.array([0, 0, -100.0])

        self.angles = {  # x, y, z
            'w': (angle_speed, 0, 0),
            's': (-angle_speed, 0, 0),
            'a': (0, -angle_speed, 0),
            'd': (0, angle_speed, 0),
            'q': (0, 0, -angle_speed),
            'e': (0, 0, angle_speed)
        }

        self.trans = { # x, y, z
            'l': (trans_speed, 0, 0),
            'j': (-trans_speed, 0, 0),
            'o': (0, trans_speed, 0),
            'u': (0, -trans_speed, 0),
            'i': (0, 0, trans_speed),
            'k': (0, 0, -trans_speed),
        }

        self.points, self.colors = self._calc_point_cloud()

        self.position = (20, 40)
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 1
        text_size, _ = cv2.getTextSize("text", self.font_face, self.font_scale,
                                       self.thickness)
        self.line_width = text_size[0]
        self.line_height = text_size[1] + 5

    def visualize(self):
        r = self.r
        t = self.t
        self._view(r, t)
        while 1:
            key = cv2.waitKey(0)

            if key not in range(256):
                continue

            ch = chr(key)
            if ch in self.trans:
                t = self.translate(t, self.trans[ch])
                self._view(r, t)

            if ch in self.angles:
                r = self.rotate(r, (-ag for ag in self.angles[ch]))
                t = self.rotate(t, self.angles[ch])
                self._view(r, t)

            if ch == 'h':  # reset
                self._view(self.r, self.t)

            elif ch == '\x1b':  # esc
                cv2.destroyAllWindows()
                break

    def _view(self, r, t):
        cv2.imshow('projected', self._calc_projected_image(r, t))


    @staticmethod
    def rotate(arr, angle):
        anglex, angley, anglez = angle
        return np.array([  # rx
            [1, 0, 0],
            [0, np.cos(anglex), -np.sin(anglex)],
            [0, np.sin(anglex), np.cos(anglex)]
        ]).dot(np.array([  # ry
            [np.cos(angley), 0, np.sin(angley)],
            [0, 1, 0],
            [-np.sin(angley), 0, np.cos(angley)]
        ]).dot(np.array([  # rz
            [np.cos(anglez), -np.sin(anglez), 0],
            [np.sin(anglez), np.cos(anglez), 0],
            [0, 0, 1]
        ]))).dot(arr)


    @staticmethod
    def translate(arr, trans):
        return arr + np.array(trans)

    @staticmethod
    def remove_invalid(disp_arr, points, colors):
        # TODO: Add nearclip to filter out the wrong projection
        mask = ((disp_arr > disp_arr.min()) & np.all(~np.isnan(points), axis=1)
                & np.all(~np.isinf(points), axis=1))
        return points[mask], colors[mask]

    def _calc_point_cloud(self):
        points = cv2.reprojectImageTo3D(self.disp,
                                        self.q,
                                        handleMissingValues=True).reshape(
                                            -1, 3)
        if self.show_disp_map:
            colors = self.disp.reshape(-1, 1)
            COL = MplColorHelper('viridis', colors.min(), colors.max())
            colors = COL.get_rgb(colors).reshape(-1, 3)
        elif self.show_depth_map:
            colors = triangulation(self.disp).reshape(-1, 1)
            COL = MplColorHelper('viridis', colors.min(), colors.max())
            colors = COL.get_rgb(colors).reshape(-1, 3)
        else:
            colors = self.img_l.reshape(-1, 3)
        return self.remove_invalid(self.disp.reshape(-1), points, colors)

    def _project_points(self, r, t):
        projected, _ = cv2.projectPoints(self.points, r, t, self.k,
                                         self.dist_coeff)
        xy = projected.reshape(-1, 2).astype(np.int)
        mask = ((0 <= xy[:, 0]) & (xy[:, 0] < self.width) & (0 <= xy[:, 1]) &
                (xy[:, 1] < self.height))
        return xy[mask], self.colors[mask]

    def _calc_projected_image(self, r, t):
        xy, cm = self._project_points(r, t)
        image = np.zeros((self.height, self.width, 3), dtype=cm.dtype)
        image[xy[:, 1], xy[:, 0]] = cm
        return image


class MplColorHelper:
    """ A helper class to create "heatmap" for a matrix """
    def __init__(self, cmap_name, min_val, max_val):
        """ constructor
            Args:
                cmap_name (str): name of cmap. ref: https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
                min_val (float): minimum value of the heatmap
                max_val (flat): maximum value of the heatmap
            Example:
                >> mat = get_depth_map()
                >> COL = MplColorHelper('viridis', 0., 1.)
                >> plt.imshow(COL.get_rgb(mat))
        """
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val, dtype='uint8'):
        out = self.scalarMap.to_rgba(val)[:, :, :3]
        if dtype == 'float':
            return out
        else:
            return np.clip(out * 255, 0, 255).astype(np.uint8)


def imread(path, channel=-1):
    """ Read raw RGB and DO NOT perform any process to the image 
        if channel <= -1: load image with original dimension
        if channel ==  0: load image with 1 channel (grey)
        if channel >=  1: load image with 3 channel (RGB) 
    """
    rgb = cv2.imread(path, channel)  # Load image with original dimension
    assert rgb is not None, "Image {} is None".format(path)
    assert rgb.ndim in [2, 3], "Image {} has ndim {}".format(path, rgb.ndim)
    #rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB) if rgb.ndim == 3 else rgb

    return rgb


def read_depth(path):
    """ Depth maps (annotated and raw Velodyne scans) are saved as uint16 PNG images,
        which can be opened with either MATLAB, libpng++ or the latest version of
        Python's pillow (from PIL import Image). A 0 value indicates an invalid pixel
        (ie, no ground truth exists, or the estimation algorithm didn't produce an
        estimate for that pixel). Otherwise, the depth for a pixel can be computed
        in meters by converting the uint16 value to float and dividing it by 256.0:

        disp(u,v)  = ((float)I(u,v))/256.0;
        valid(u,v) = I(u,v)>0;
    """
    depth = imread(path).astype(np.float32) / 256.0
    assert depth.ndim == 2, f"Depth from {path} has more ndim ({depth.ndim}) than 2Î"
    return depth


def triangulation(disp):
    # Get depth from disp
    baseline = 0.54
    focal_length = 721.5377
    depth = baseline * focal_length / disp
    depth[disp <= 1e-4] = 0
    depth = 255 * (depth - depth.min()) / depth.max()
    print("Min {}, MAX {}".format(depth.min(), depth.max()))
    return depth.astype(np.uint8)


def main():
    args = parse_args()

    img_l = imread(args.left_image_name, 1)
    img_r = imread(args.right_image_name, 1)

    # Init SGBM
    sgm = SGBMatcher(num_disparities=args.num_disparities,
                     block_size=args.block_size,
                     window_size=args.window_size)

    sgm.show_param()

    # Get disparity
    disp = sgm.compute(img_l, img_r)

    # Sparsify the computated points
    disp_ = np.zeros_like(disp)
    disp_[::args.M, ::args.N] = disp[::args.M, ::args.N]

    # Init Visualizer
    rec = Reconstructor(img_l=img_l,
                        img_r=img_r,
                        disp=disp,
                        focal_length=args.focal_length,
                        baseline=args.distance_between_cameras,
                        show_disp_map=args.show_disp_map,
                        show_depth_map=args.show_depth_map)

    rec.visualize()


if __name__ == '__main__':
    main()
