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
"""
Simple loader of scannet for evaluation and gt tsdf generation.
"""

import os

import numpy as np
import cv2


class ScanNetDataset:
    """
    A class that reads and parses the scannet dataset for a single scene.

    Args:
        n_imgs (int): Number of color images of a single scene.
        scene (str): Scene name of the scannet dataset.
        data_path (str): Path to the scannet dataset.
        max_depth (bool): Depth values filter threshold.
        id_list (list[int]): Id list of color images. Default: None.

    Examples:
        >>> from src.tools.simple_loader import ScanNetDataset
        >>> dataset_generator = ScanNetDataset(n_imgs, scene, data_path, max_depth)
    """

    def __init__(self, n_imgs, scene, data_path, max_depth, id_list=None):
        self.n_imgs = n_imgs
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        if id_list is None:
            self.id_list = [i for i in range(n_imgs)]
        else:
            self.id_list = id_list

    def __len__(self):
        """
        Return the length of the dataset.
        """

        return self.n_imgs

    def __getitem__(self, idx):
        """
        Loads individual frames by id, and return cam_pose, depth_img, color_img.

        Args:
            idx (int): The number of images.

        Returns:
            numpy.ndarray, the camera pose.
            numpy.ndarray, the depth image.
            numpy.ndarray, the color image.

        Examples:
            >>> (cam_pose, depth_img, color_img) = dataset[idx]
        """

        idx = self.id_list[idx]
        cam_pose = np.loadtxt(os.path.join(self.data_path, self.scene, "pose", str(idx) + ".txt"), delimiter=' ')
        # Read depth image and camera pose
        depth_img = cv2.imread(os.path.join(self.data_path, self.scene, "depth", str(idx) + ".png"), -1).astype(
            np.float32)
        depth_img /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_img[depth_img > self.max_depth] = 0
        # Read RGB image
        color_img = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, self.scene, "color", str(idx) + ".jpg")),
                                 cv2.COLOR_BGR2RGB)
        color_img = cv2.resize(color_img, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_AREA)
        return cam_pose, depth_img, color_img
