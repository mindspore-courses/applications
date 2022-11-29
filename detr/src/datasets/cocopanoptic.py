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
# ======================================================================
"""Build COCO_panoptic dataset."""
import json
import os

import numpy as np
from PIL import Image
import mindspore.dataset as ds
from panopticapi.utils import rgb2id

from .transforms import make_transforms, batch_fn


class COCOPanoptic:
    """
    A class for COCO panoramic segmentation dataset.

    Args:
        img_folder (str): The folder path of COCO images.
        ann_folder (str): The folder path of COCO panoramic segmentation images.
        ann_file (str): Path of COCO panoramic segmentation annotations file.
        return_masks (bool, optional): If true, remove images without annotations.

    Returns:
        dataset.

    Examples:
        >>> dataset = COCOPanoptic('coco/val2017', 'coco_panoptic/panoptic_val2017',
                                   'coco_panoptic/annotations/panoptic_val2017.json')
    """

    def __init__(self, img_folder, ann_folder, ann_file, return_masks=True):
        """Initialize the dataset class."""
        with open(ann_file, 'r', encoding='utf-8') as f:
            self.coco = json.load(f)
        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])
        if "annotations" in self.coco:
            for img, ann in zip(self.coco['images'], self.coco['annotations']):
                assert img['file_name'][:-4] == ann['file_name'][:-4]
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.return_masks = return_masks

    def __getitem__(self, idx):
        """ Get a list of datasets """
        if "annotations" in self.coco:
            ann_info = self.coco['annotations'][idx]
        else:
            ann_info = self.coco['images'][idx]
        img_path = os.path.join(self.img_folder, ann_info['file_name'].replace('.png', '.jpg'))
        ann_path = os.path.join(self.ann_folder, ann_info['file_name'])
        img = (Image.open(img_path).convert('RGB'))
        w, h = img.size

        if "segments_info" in ann_info:
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
            masks = rgb2id(masks)
            ids = np.array([ann['id'] for ann in ann_info['segments_info']])
            masks = masks == ids[:, None, None]
            labels = [ann['category_id'] for ann in ann_info['segments_info']]

        target = {}
        target['image_id'] = [ann_info['image_id'] if "image_id" in ann_info else ann_info["id"]]
        if self.return_masks:
            target['masks'] = masks
        target['labels'] = labels
        target["boxes"] = _masks_to_boxes(masks)
        target['size'] = [int(h), int(w)]
        target['orig_size'] = [int(h), int(w)]

        if "segments_info" in ann_info:
            for name in ['iscrowd', 'area']:
                target[name] = [ann[name] for ann in ann_info['segments_info']]
        img = np.asarray(img)
        return img, target['image_id'], target['masks'], target['labels'], target["boxes"],\
            target['size'], target['orig_size'], target['iscrowd'], target['area']

    def __len__(self):
        """ Get the number of images. """
        return len(self.coco['images'])


def _masks_to_boxes(mask):
    """
    Generate the corresponding box according to the mask.

    Args:
        mask (numpy.ndarray): Image mask with shape class_num*H*W.

    Returns:
        box (list): A list of bounding box.
    """
    h, w = mask.shape[-2:]
    y = np.arange(0, h)
    x = np.arange(0, w)
    x, y = np.meshgrid(x, y)
    x_mask = np.expand_dims(x, 0) * mask
    y_mask = np.expand_dims(y, 0) * mask
    x_max = []
    for i in x_mask:
        x_max.append(i.max())
    bb = np.ones((h, w)) * 10000
    x_min = []
    for i in np.expand_dims(bb, 0) * ~mask + x_mask:
        x_min.append(i.min())
    y_max = []
    for i in y_mask:
        y_max.append(i.max())
    y_min = []
    for i in np.expand_dims(bb, 0) * ~mask + y_mask:
        y_min.append(i.min())
    box = []
    for i in range(len(x_min)):
        box.append([x_min[i], y_min[i], x_max[i], y_max[i]])
    return box


def build(img_set='val', batch=2, shuffle=False, coco_dir='./coco', pano_dir='./coco_panoptic'):
    """
    Build COCO panoramic segmentation dataset

    Args:
        img_set (str): Set dataset for train or val.
        batch (int): Batch of dataset.
        shuffle (bool): Shuffle data if True.
        COCO_dir (str): The folder path of COCO images.
        pano_dir (str): The folder path of COCO panoramic segmentation images.

    Returns:
        COCO panoptic dataset.

    Examples:
        >>> dataset = build(img_set='val', batch=2, shuffle=False, coco_dir='./coco', pano_dir='./coco_panoptic')
    """
    p_json_path = f"{pano_dir}/annotations/panoptic_{img_set}2017.json"
    img_dir = f"{coco_dir}/{img_set}2017"
    p_img_dir = f"{pano_dir}/panoptic_{img_set}2017"
    target_list = ['img', 'image_id', 'masks', 'cats', 'bbox',
                   'size', 'orig_size', 'iscrowd', 'area']
    out_list = ["img", 'img_mask', 'img_id', 'masks', 'cats',
                'bbox', 'size', 'orig_size', 'iscrowd', 'area', 'len_list']
    dataset_generator = COCOPanoptic(img_dir, p_img_dir, p_json_path)
    dataset = ds.GeneratorDataset(dataset_generator, target_list, shuffle=shuffle)
    dataset = make_transforms(img_set, dataset)
    dataset = dataset.batch(batch, drop_remainder=False, input_columns=target_list,
                            output_columns=out_list, per_batch_map=batch_fn)
    return dataset
