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
"""COCO Dataset"""
import os
import json

import numpy as np
from pycocotools.coco import COCO

from src.process_datasets.joints_dataset import JointsDataset


class COCODataset(JointsDataset):
    """
    COCO Dataset for MSPN Model

    Args:
        keypoint_num (int): Num of Keypoints.
        flip_pairs (list): Keypoints Pairs Index.
        upper_body_ids (list): Upper Body Keypoints Index.
        lower_body_ids (list): Lower Body Keypoints Index.
        input_shape (list): Input Image Shape.
        output_shape (list): Output Image Shape.
        stage (str): Train or Eval or Test Stage.
        data_dir (str): Image Data Directory Path.
        det_file_path (str): Detection JSON File Path. Default: None.
        gt_file_path (str): Grount Truth Label JSON File Path. Default: None.

    Examples:
        >>> dataset = COCODataset(17, [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], \
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [11, 12, 13, 14, 15, 16], [256, 192], [64, 48], 'train', \
        './coco2014', './annotation/det.json', './annotation/gt.json')
    """
    def __init__(self,
                 keypoint_num: int,
                 flip_pairs: list,
                 upper_body_ids: list,
                 lower_body_ids: list,
                 input_shape: list,
                 output_shape: list,
                 stage: str,
                 data_dir: str,
                 det_file_path: str = None,
                 gt_file_path: str = None
                 ) -> None:
        super().__init__(keypoint_num, flip_pairs, upper_body_ids, lower_body_ids, input_shape, output_shape, stage)
        self._exception_ids = ['366379']
        self.cur_dir = os.path.split(os.path.realpath(__file__))[0]
        self.img_dir = data_dir
        self.gt_file_path = gt_file_path
        self.det_file_path = det_file_path
        self.data = self._get_data()
        self.data_num = len(self.data)

    def _get_data(self):
        """Data Preprocess"""
        data = list()
        if self.gt_file_path:
            coco = COCO(self.gt_file_path)
        if self.stage == 'train':
            for aid, ann in coco.anns.items():
                img_id = ann['image_id']
                if img_id not in coco.imgs \
                        or img_id in self._exception_ids:
                    continue

                if ann['iscrowd']:
                    continue

                img_name = coco.imgs[img_id]['file_name']
                prefix = 'val2014' if 'val' in img_name else 'train2014'
                img_path = os.path.join(self.img_dir, prefix, img_name)

                bbox = np.array(ann['bbox'])
                area = ann['area']
                joints = np.array(ann['keypoints']).reshape((-1, 3))
                head_rect = np.array([0, 0, 1, 1], np.int32)
                center, scale = self._bbox_to_center_and_scale(bbox)
                if np.sum(joints[:, -1] > 0) < self.kp_load_min_num or ann['num_keypoints'] == 0:
                    continue

                d = dict(aid=aid,
                         area=area,
                         bbox=bbox,
                         center=center,
                         headRect=head_rect,
                         img_id=img_id,
                         img_name=img_name,
                         img_path=img_path,
                         joints=joints,
                         scale=scale)

                data.append(d)
        else:
            det_path = self.det_file_path
            dets = json.load(open(det_path))
            for det in dets:
                if self.stage == 'val' and (det['image_id'] not in coco.imgs or det['category_id'] != 1):
                    continue

                img_id = det['image_id']
                if self.stage == 'val':
                    img_name = 'COCO_val2014_000000%06d.jpg' % img_id
                    img_path = os.path.join(self.img_dir, 'val2014', img_name)
                else:
                    img_name = str(img_id) + '.jpg'
                    img_path = os.path.join(self.img_dir, img_name)

                bbox = np.array(det['bbox'])
                center, scale = self._bbox_to_center_and_scale(bbox)
                joints = np.zeros((self.keypoint_num, 3))
                score = det['score']
                head_rect = np.array([0, 0, 1, 1], np.int32)

                d = dict(bbox=bbox,
                         center=center,
                         headRect=head_rect,
                         img_id=img_id,
                         img_name=img_name,
                         img_path=img_path,
                         joints=joints,
                         scale=scale,
                         score=score)

                data.append(d)

        return data

    def _bbox_to_center_and_scale(self, bbox):
        """Convert Coordinate"""
        x, y, w, h = bbox

        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w / 2.0
        center[1] = y + h / 2.0

        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32)

        return center, scale
