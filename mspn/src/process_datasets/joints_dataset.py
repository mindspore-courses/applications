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
# ==============================================================================
"""Joints Dataset"""
import copy
import secrets

import cv2
import numpy as np

from src.utils.keypoints_transforms import get_affine_transform, affine_transform, flip_joints


class JointsDataset:
    """
    Joints Dataset

    Args:
        keypoint_num (int): Number of Human Keypoints for MSPN.
        flip_pairs (list): Human Keypoints Pairs Index.
        upper_body_ids (list): Upper Body Keypoints Index.
        lower_body_ids (list): Lower Body Keypoints Index.
        input_shape (list): Input Image Shape.
        output_shape (list): Output Image Shape.
        stage (str): Train or Eval or Test Stage.

    Examples:
        >>> dataset = JointsDataset(17, [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], \
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [11, 12, 13, 14, 15, 16], [256, 192], [64, 48], 'train')
    """
    def __init__(self,
                 keypoint_num: int,
                 flip_pairs: list,
                 upper_body_ids: list,
                 lower_body_ids: list,
                 input_shape: list,
                 output_shape: list,
                 stage: str
                 ) -> None:
        self.stage = stage
        if self.stage not in ('train', 'val', 'test'):
            raise ValueError("Stage Argument Should Be Chosen From ['train', 'val', 'test']")
        self.data = list()

        # In-paras
        self.keypoint_num = keypoint_num
        self.flip_pairs = flip_pairs
        self.upper_body_ids = upper_body_ids
        self.lower_body_ids = lower_body_ids
        self.kp_load_min_num = 1
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.w_h_ratio = self.input_shape[1] / self.input_shape[0]

        # Aug-paras
        self.pixel_std = 200
        self.basic_ext = 0.05
        self.rand_ext = True
        self.x_ext = 0.6
        self.y_ext = 0.8
        self.scale_factor_low = -0.25
        self.scale_factor_high = 0.25
        self.scale_shrink_ratio = 0.8
        self.rotation_factor = 45
        self.prob_rotation = 0.5
        self.prob_flip = 0.5
        self.num_keypoints_half_body = 3
        self.prob_half_body = 0.3
        self.x_ext_half_body = 0.6
        self.y_ext_half_body = 0.8
        self.add_more_aug = False
        self.gaussian_kernels = [[15, 15], [11, 11], [9, 9], [7, 7], [5, 5]]
        self.test_x_ext = 0.09
        self.test_y_ext = 0.135

    def __len__(self):
        """Num of Total Data"""
        return self.data_num

    def __getitem__(self, idx):
        """Get Item with specific Index"""
        d = copy.deepcopy(self.data[idx])
        img_id = d['img_id']
        img_path = d['img_path']
        data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR)
        joints = d['joints'][:, :2]
        joints_vis = d['joints'][:, -1].reshape((-1, 1))
        center = d['center']
        scale = d['scale']
        score = d['score'] if 'score' in d else 1
        rotation = 0

        if self.stage == 'train':
            scale[0] *= (1 + self.basic_ext)
            scale[1] *= (1 + self.basic_ext)
            rand_1 = 1.0
            rand_2 = 1.0
            if self.rand_ext:
                rand_1 = np.random.rand()
                rand_2 = np.random.rand()
            scale[0] *= (1 + rand_1 * self.x_ext)
            scale[1] *= (1 + rand_2 * self.y_ext)
        else:
            scale[0] *= (1 + self.test_x_ext)
            scale[1] *= (1 + self.test_y_ext)

        # fit the ratio
        if scale[0] > self.w_h_ratio * scale[1]:
            scale[1] = scale[0] * 1.0 / self.w_h_ratio
        else:
            scale[0] = scale[1] * 1.0 * self.w_h_ratio

        # augmentation
        if self.stage == 'train':
            # half body
            if np.sum(joints_vis[:, 0] > 0) > self.num_keypoints_half_body and np.random.rand() < self.prob_half_body:
                center, scale = self.half_body_transform(joints, joints_vis, center, scale)

            # scale
            rand = secrets.randbelow(int((self.scale_factor_high - self.scale_factor_low) * 100)) / 100
            rand += self.scale_factor_low + 1
            scale_ratio = self.scale_shrink_ratio * rand
            scale *= scale_ratio

            # rotation
            if secrets.randbelow(100) / 100 <= self.prob_rotation:
                rotation = secrets.randbelow(2 * self.rotation_factor * 100) / 100 - self.rotation_factor

            # flip
            if secrets.randbelow(100) / 100 <= self.prob_flip:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = flip_joints(joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        trans = get_affine_transform(center, scale, rotation, self.input_shape)
        img = cv2.warpAffine(data_numpy, trans, (int(self.input_shape[1]), int(self.input_shape[0])),
                             flags=cv2.INTER_LINEAR)

        if self.stage == 'train':
            for i in range(self.keypoint_num):
                if joints_vis[i, 0] > 0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                    if joints[i, 0] < 0 \
                            or joints[i, 0] > self.input_shape[1] - 1 \
                            or joints[i, 1] < 0 \
                            or joints[i, 1] > self.input_shape[0] - 1:
                        joints_vis[i, 0] = 0
            valid = np.array(joints_vis, dtype=np.float32)

            labels_num = len(self.gaussian_kernels)
            labels = np.zeros((labels_num, self.keypoint_num, *self.output_shape))
            for i in range(labels_num):
                labels[i] = self.generate_heatmap(joints, valid, kernel=self.gaussian_kernels[i])
            labels = np.array(labels, dtype=np.float32)

            return img, valid, labels

        return img, score, center, scale, img_id

    def _get_data(self):
        """Data Preprocess"""
        raise NotImplementedError

    def evaluate(self, pred_path):
        """Evaluate Results"""
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis, pre_center, pre_scale):
        """Half Body Transform Augmentation for Joints"""
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.keypoint_num):
            if joints_vis[joint_id, 0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 3:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints if len(lower_joints) > 3 else upper_joints

        if len(selected_joints) < 3:
            return pre_center, pre_scale

        selected_joints = np.array(selected_joints, dtype=np.float32)

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        center = (left_top + right_bottom) / 2

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        rand = np.random.rand()
        w *= (1 + rand * self.x_ext_half_body)
        rand = np.random.rand()
        h *= (1 + rand * self.y_ext_half_body)

        if w > self.w_h_ratio * h:
            h = w * 1.0 / self.w_h_ratio
        elif w < self.w_h_ratio * h:
            w = h * self.w_h_ratio

        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32)

        return center, scale

    def generate_heatmap(self, joints, valid, kernel=(7, 7)):
        """Generate Coarse-to-Fine Supervision Label"""
        heatmaps = np.zeros((self.keypoint_num, *self.output_shape), dtype='float32')

        for i in range(self.keypoint_num):
            if valid[i] < 1:
                continue
            target_y = joints[i, 1] * self.output_shape[0] / self.input_shape[0]
            target_x = joints[i, 0] * self.output_shape[1] / self.input_shape[1]
            heatmaps[i, int(target_y), int(target_x)] = 1
            heatmaps[i] = cv2.GaussianBlur(heatmaps[i], kernel, 0)
            maxi = np.amax(heatmaps[i])
            if maxi <= 1e-8:
                continue
            heatmaps[i] /= maxi / 255

        return heatmaps
