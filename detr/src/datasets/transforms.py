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
"""data transforms api"""
import random

import cv2
import numpy as np
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision.c_transforms as c_vision


__all__ = ['make_transforms', 'batch_fn']


def resize(image, target, size, max_size=1333):
    """
    Change the size of image, boxes, area, size and masks

    Args:
        image (numpy.ndarray): Image array.
        target (dict): A dict with element of boxes, area, size, masks.
        size (int): Target size to be transformed.
        max_size (int, optional): Maximum size for transformation. Defaults to 800.

    Returns:
        rescaled_image (numpy.ndarray): Image array after resize.
        target (dict): A dict with element of boxes, area, size, masks.
    """
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=max_size):
        h, w = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return oh, ow

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.shape[:2], size, max_size)
    resize_op = c_vision.Resize(size, Inter.LINEAR)
    rescaled_image = resize_op(image)

    if target is None:
        return rescaled_image, None
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.shape[:2], image.shape[:2]))
    ratio_width, ratio_height = ratios
    target = target.copy()

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area
    h, w = size
    target["size"] = np.array([h, w])

    if "masks" in target:
        masks = target['masks'].astype(np.float32)
        resize_op = c_vision.Resize(size, Inter.NEAREST)
        target['masks'] = resize_op(masks.transpose((1, 2, 0))).transpose((2, 0, 1)) > 0.5

    return rescaled_image, target


class RandomResize():
    """
    Class of random resize.

    Args:
        sizes (list): A list of sizes for transformation.
        max_size (int, optional): Maximum size for transformation. Defaults to 800.
    """

    def __init__(self, sizes, max_size=1333):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, bbox, size, area, masks):
        size = random.choice(self.sizes)
        target = {'bbox': bbox, 'size': size, 'area': area, 'masks': masks}
        img, target = resize(img, target, size, self.max_size)
        bbox = target['bbox']
        size = target['size']
        area = target['area']
        masks = target['masks']
        return img, bbox, size, area, masks


class RandomHorizontalFlip():
    """
    Class of random horizontal flip

    Args:
        prob (float, optional): Probability of horizontal flip. Defaults to 0.5.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, bbox, masks):
        if random.random() < self.prob:
            return hflip(img, bbox, masks)
        return img, bbox, masks


def hflip(img, bbox, masks):
    """
    Flip horizontally of img, bbox and masks

    Args:
        img (numpy.ndarray): A numpy array of shape [H, W, 3].
        bbox (numpy.ndarray): A numpy array of shape [num, 4].
        masks (numpy.ndarray): A numpy array of shape [num,H,W].

    Returns:
        img (numpy.ndarray): A numpy array of shape [H, W, 3].
        bbox (numpy.ndarray): A numpy array of shape [num, 4].
        masks (numpy.ndarray): A numpy array of shape [num,H,W].
    """
    img = cv2.flip(img, 1)
    bbox = bbox[:, [0, 1, 2, 3]] * np.array([-1, 1, 1, 1]) + np.array([1, 0, 0, 0])
    masks = np.flip(masks, -1)
    return img, bbox, masks


def make_transforms(image_set, dataset):
    """
    Data augmentation and normalization

    Args:
        image_set (str): Set dataset for train or val
        dataset (class): Dataset for augmentation and normalization

    Returns:
        dataset: Dataset after augmentation and normalization
    """
    scales = [480, 512, 544, 576, 608, 640]
    in_list = ["img", 'bbox', 'size', 'area', 'masks']

    if image_set == 'train':
        randomhorizontalflip_op = RandomHorizontalFlip()
        randomresize_op = RandomResize(scales)
        normalize_op = c_vision.Normalize([123.7, 116.3, 103.5], [58.4, 57.1, 57.4])
        dataset = dataset.map(randomhorizontalflip_op, input_columns=['img', 'bbox', 'masks'])
        dataset = dataset.map(randomresize_op, input_columns=in_list, num_parallel_workers=8)
        dataset = dataset.map(normalize_op, input_columns=['img'])
        return dataset

    if image_set == 'val':
        randomresize_op = RandomResize([800])
        normalize_op = c_vision.Normalize([123.7, 116.3, 103.5], [58.4, 57.1, 57.4])
        dataset = dataset.map(randomresize_op, input_columns=in_list, num_parallel_workers=8)
        dataset = dataset.map(normalize_op, input_columns=['img'])
        return dataset
    return None


def _cat_expand_repeat(x, bs):
    """
    Concatenate, expand and repeat data.

    Args:
        x (list): A list of numpy array.
        bs (int): Batch of dataset.

    Returns:
        numpy.ndarray: Batch numpy array.
    """
    return np.repeat(np.expand_dims(np.concatenate(x), 0), bs, 0)


def _max_by_axis(the_list):
    """
    Get the maximum width and height in the batch images.

    Args:
        the_list (list): A list of images shape.

    Returns:
        list: Maximum image shape.
    """
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def batch_fn(img, img_id, masks, cats, bbox, size, orig_size, iscrowd, area, _):
    """
    Batch processing of data

    Args:
        img (list): A list of images, and each element is an array of shape H*W*3.
        img_id (list): A list of images id, and each element is an array of length 1.
        masks (list): A list of images masks, and each element is an array of shape class_num*H*W.
        cats (list): A list of categories, and each element is an array of length class_num.
        bbox (list): A list of bbox, and each element is an array of shape class_num*4.
        size (list): A list of images size, and each element is an array of length 2.
        orig_size (list): A list of images original size, and each element is an array of length 2.
        iscrowd (list): A list of iscrowd type, and each element is an array of length class_num.
        area (list): A list of mask area, and each element is an array of length class_num.

    Returns:
        img (numpy.ndarray): Batch images with shape N*3*H*W.
        masks (numpy.ndarray): Batch images masks with shape N*class_num*H*W.
        img_id (numpy.ndarray): Batch image ids with shape N*class_num.
        cats (numpy.ndarray): Batch categories with shape N*class_num.
        bbox (numpy.ndarray): Batch bbox with shape N*class_num*4.
        size (numpy.ndarray): Batch images size with shape N*2.
        orig_size (numpy.ndarray): Batch images original size with shape N*2.
        iscrowd (numpy.ndarray): Batch iscrowd type with shape N*class_num.
        area (numpy.ndarray): Batch mask area with shape N*class_num.
        len_list (numpy.ndarray): Batch objects with shape N*1.
    """
    for i, v in enumerate(img):
        img[i] = v.transpose((2, 0, 1))
    max_size = _max_by_axis([list(im.shape) for im in img])
    batch_shape = [len(img)] + max_size
    bs, _, h, w = batch_shape
    img_p = np.zeros(batch_shape).astype(np.float32)
    img_mask = np.ones((bs, h, w)).astype(np.float32)
    for i in range(bs):
        img_p[i][: img[i].shape[0], : img[i].shape[1], : img[i].shape[2]] = img[i]
        img_mask[i][: img[i].shape[1], :img[i].shape[2]] = 0

    len_list = []
    masks_cat = []
    cats_cat = []
    bbox_cat = []
    iscrowd_cat = []
    area_cat = []
    for i in range(bs):
        num = masks[i].shape[0]
        len_list.append([num])
        masks_p = np.zeros((num, h, w)).astype(np.float32)
        masks_p[: masks[i].shape[0], : masks[i].shape[1], : masks[i].shape[2]] = (masks[i])
        masks_cat.append(masks_p)
        cats_cat.append(cats[i])
        bbox_cat.append(bbox[i])
        iscrowd_cat.append(iscrowd[i])
        area_cat.append(area[i])

    masks = _cat_expand_repeat(masks_cat, bs)
    cats = _cat_expand_repeat(cats_cat, bs)
    bbox = _cat_expand_repeat(bbox_cat, bs)
    iscrowd = _cat_expand_repeat(iscrowd_cat, bs)
    area = _cat_expand_repeat(area_cat, bs)
    return img_p, img_mask, img_id, masks, cats, bbox, size, orig_size, iscrowd, area, len_list
