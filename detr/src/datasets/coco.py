# Copyright 2022 Huawei Technologies Co., Ltdmake_transforms
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
"""Build COCO dataset."""
import os

import numpy as np
from PIL import Image
import mindspore.dataset as ds
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from .transforms import make_transforms, batch_fn


def _has_only_empty_bbox(anno):
    """
    Check empty bbox.

    Args:
        anno (list): A list of COCO annotation.

    Returns:
        bool: True if annotation has empyty bbox.
    """
    return all(any(ob <= 1 for ob in obj["bbox"][2:]) for obj in anno)


def _has_valid_annotation(anno):
    """
    Check annotation file.

    Args:
        anno (list): A list of COCO annotation.

    Returns:
        bool: True if annotation has valid annotation.
    """
    # if it's empty, there is no annotation
    if not anno:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different criteria for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    return False


def _box_xywh_to_cxcywh(bbox, w_orig, h_orig):
    """
    Convert xywh to cxcyhw.

    Args:
        bbox (list): A list of box coordinate[x, y, w, h].
        w_orig (int): Original width.
        h_orig (int): Original height.

    Returns:
        list: A list of box coordinate[center_x, center_y, w, h].
    """
    x0, y0, w_cur, h_cur = bbox
    return [(x0 + w_cur / 2) / w_orig, (y0 + h_cur / 2) / h_orig, w_cur / w_orig, h_cur / h_orig]


class COCODataset:
    """
    A class for COCO dataset.

    Args:
        img_path (str): The folder path of COCO images.
        json_path (str): Path of COCO annotations file.
        remove_images_without_annotations (bool, optional): If true, remove images without annotations.

    Returns:
        dataset.

    Examples:
        >>> dataset = COCODataset('coco/val2017', 'coco/annotations/instances_val2017.json')
    """

    def __init__(self, img_path, json_path, remove_images_without_annotations=True):
        """ Initialize the dataset class. """
        self.coco = COCO(annotation_file=json_path)
        self.img_path = img_path
        self.imgs_ids = list(sorted(self.coco.imgs.keys()))
        # filter images without any annotations
        if remove_images_without_annotations:
            img_ids = []
            for img_id in self.imgs_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if _has_valid_annotation(anno):
                    img_ids.append(img_id)
            self.imgs_ids = img_ids

    def __getitem__(self, index):
        """ Get a list of datasets. """
        img_id = self.imgs_ids[index]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        file_path = os.path.join(self.img_path, file_name)
        img = Image.open(file_path).convert('RGB')
        img = np.array(img).astype(np.float32)
        h, w = img.shape[:2]
        ann_ids = self.coco.getAnnIds(img_id)
        target = self.coco.loadAnns(ann_ids)
        target = [obj for obj in target if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        # Get labels
        seg = []
        seg_len = []
        bbox = []
        cats = []
        area = []
        iscrowd = []
        masks = []

        for i in target:
            seg += i['segmentation'][0]
            seg_len.append(len(i['segmentation'][0]))
            rles = coco_mask.frPyObjects([i['segmentation'][0]], img.shape[0], img.shape[1])
            mask = coco_mask.decode(rles)[:, :, 0]
            masks.append(mask)
            box = _box_xywh_to_cxcywh(i['bbox'], w, h)
            bbox.append(box)
            cats.append(i['category_id'])
            area.append(i['area'])
            iscrowd.append(i['iscrowd'])

        img_id = [img_id]
        size = [img.shape[0], img.shape[1]]
        orig_size = [img.shape[0], img.shape[1]]
        bbox = np.array(bbox).astype(np.float32)
        return img, img_id, masks, cats, bbox, size, orig_size, iscrowd, area

    def __len__(self):
        """ Get the number of images. """
        return len(self.imgs_ids)


def build(img_set='val', batch=2, shuffle=False, coco_dir='./coco'):
    """
    Build COCO dataset

    Args:
        img_set (str): Set dataset for train or val
        batch (int): Batch of dataset
        shuffle (bool): Shuffle data if true
        coco_dir (str): Path of COCO data

    Return:
        COCO dataset

    Examples:
        >>> dataset = build(img_set='val', batch=2, shuffle=False, coco_dir='./coco')
    """
    img_path = f"{coco_dir}/{img_set}2017"
    json_path = f"{coco_dir}/annotations/instances_{img_set}2017.json"

    target_list = ["img", 'img_id', 'masks', 'cats', 'bbox', 'size', 'orig_size', 'iscrowd', 'area']
    out_list = ["img", 'img_mask', 'img_id', 'masks', 'cats',
                'bbox', 'size', 'orig_size', 'iscrowd', 'area', 'len_list']
    dataset_generator = COCODataset(img_path, json_path)
    dataset = ds.GeneratorDataset(dataset_generator, target_list, shuffle=shuffle)
    dataset = make_transforms(img_set, dataset)
    dataset = dataset.batch(batch, drop_remainder=False, input_columns=target_list,
                            output_columns=out_list, per_batch_map=batch_fn)
    return dataset
