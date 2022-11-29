# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Smooth the animation image and save it in a new directory."""

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    """Argument parsing"""
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--style_path', default='/home/nc438/ztf/animeganv2/dataset/Sakura/smooth', type=str)
    parser.add_argument('--output_path', default='/home/nc438/ztf/animeganv2/dataset/Sakura/style', type=str)
    return parser.parse_args()


def make_edge_smooth(style_dir, output_dir, img_size=256):
    """
    Generate images with smooth edges.

    Args:
        style_dir (str): Style image path.
        output_dir (str): Output image path.
        img_size (int): Image size. Default: 256.
    """

    file_list = os.listdir(style_dir)
    save_dir = output_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    for f in tqdm(file_list):

        bgr_img = cv2.imread(os.path.join(style_dir, f))
        gray_img = cv2.imread(os.path.join(style_dir, f), 0)

        bgr_img = cv2.resize(bgr_img, (img_size, img_size))
        pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (img_size, img_size))

        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(bgr_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        cv2.imwrite(os.path.join(save_dir, f), gauss_img)


if __name__ == '__main__':
    args = parse_args()
    make_edge_smooth(args.style_path, args.output_path, img_size=256)
