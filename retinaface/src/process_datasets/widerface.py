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
""" Create the WiderFace dataset. """

import os
import copy

import cv2
import numpy as np
from mindspore import dataset as de

from process_datasets.pre_process import PreProcessor
from utils.detection import BboxEncoder


class WiderFace:
    """WiderFace"""

    def __init__(self, label_path):
        self.images_list = []
        self.labels_list = []
        f = open(label_path, 'r')
        lines = f.readlines()
        first = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if first is True:
                    first = False
                else:
                    c_labels = copy.deepcopy(labels)
                    self.labels_list.append(c_labels)
                    labels.clear()
                # remove '# '
                path = line[2:]
                path = label_path.replace('label.txt', 'images/') + path

                assert os.path.exists(path), 'image path is not exists.'

                self.images_list.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        # add the last label
        self.labels_list.append(labels)

        # del bbox which width is zero or height is zero
        for i in range(len(self.labels_list) - 1, -1, -1):
            labels = self.labels_list[i]
            for j in range(len(labels) - 1, -1, -1):
                label = labels[j]
                if label[2] <= 0 or label[3] <= 0:
                    labels.pop(j)
            if not labels:
                self.images_list.pop(i)
                self.labels_list.pop(i)
            else:
                self.labels_list[i] = labels

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, item):
        return self.images_list[item], self.labels_list[item]


def read_dataset(img_path, annotation):
    """read_dataset"""
    cv2.setNumThreads(2)

    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = cv2.imread(img_path.tostring().decode("utf-8"))

    labels = annotation
    anns = np.zeros((0, 15))
    if labels.shape[0] <= 0:
        return anns
    for _, label in enumerate(labels):
        ann = np.zeros((1, 15))

        # get bbox
        ann[0, 0:2] = label[0:2]  # x1, y1
        ann[0, 2:4] = label[0:2] + label[2:4]  # x2, y2

        # get landmarks
        ann[0, 4:14] = label[[4, 5, 7, 8, 10, 11, 13, 14, 16, 17]]

        # set flag
        if (ann[0, 4] < 0):
            ann[0, 14] = -1
        else:
            ann[0, 14] = 1

        anns = np.append(anns, ann, axis=0)
    target = np.array(anns).astype(np.float32)

    return img, target


def create_dataset(data_dir, cfg, batch_size=32, repeat_num=1, shuffle=True, multiprocessing=True, num_worker=16,
                   num_shards=1, shard_id=0):
    """create_dataset"""
    dataset = WiderFace(data_dir)
    if num_shards == 1:
        de_dataset = de.GeneratorDataset(dataset, ["image", "annotation"],
                                         shuffle=shuffle,
                                         num_parallel_workers=num_worker)
    else:
        de_dataset = de.GeneratorDataset(dataset, ["image", "annotation"],
                                         shuffle=shuffle,
                                         num_parallel_workers=num_worker,
                                         num_shards=num_shards,
                                         shard_id=shard_id)

    aug = PreProcessor(cfg['image_size'])
    encode = BboxEncoder(cfg)

    def union_data(image, annot):
        i, a = read_dataset(image, annot)
        i, a = aug(i, a)
        out = encode(i, a)

        return out

    de_dataset = de_dataset.map(input_columns=["image", "annotation"],
                                output_columns=["image", "truths", "conf", "landm"],
                                column_order=["image", "truths", "conf", "landm"],
                                operations=union_data,
                                python_multiprocessing=multiprocessing,
                                num_parallel_workers=num_worker)
    de_dataset = de_dataset.batch(batch_size, drop_remainder=True)
    de_dataset = de_dataset.repeat(repeat_num)

    return de_dataset


def image_transform(img, val_origin_size, size1, size2):
    """
    Transform image into a specific size in order to fit the network.

    Args:
        img (numpy.ndrray): Numpy.ndarray of images, usually get from cv2.imread, a [H,W,C] shape array.
        val_origin_size (bool): Whether to evaluate the origin size image, if True, all images will be fill to the same
            size as the input size, size1 is the height and size2 is the width of specific size.If False, size1 will be
            a target size, size2 will be the max size, image will first resize to target size, if height or width of the
             image is too large, it will be then resize to max_size.
        size1 (int): If val_origin_size is True, it will be the target height of image, else it will be the target size
        of image.
        size2 (int): If val_origin_size is True, it will be the target width of image, else it will be the max size
        of image.

    Returns:
        A tuple, its first element is the image after resize, its second element is the multiple of resize.

    Raises:
        RuntimeError: If the height and width of input images do not meet requirements.
    """
    if val_origin_size:
        h_max, w_max = size1, size2
        resize = 1
        if not (img.shape[0] <= h_max and img.shape[1] <= w_max):
            raise RuntimeError('The height and width of input images do not meet requirements.')
        image_t = np.empty((h_max, w_max, 3), dtype=img.dtype)
        image_t[:, :] = (104.0, 117.0, 123.0)
        image_t[0:img.shape[0], 0:img.shape[1]] = img
        img = image_t
    else:
        target_size, max_size = size1, size2
        im_size_min = np.min(img.shape[0:2])
        im_size_max = np.max(img.shape[0:2])
        resize = float(target_size) / float(im_size_min)
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        if not (img.shape[0] <= max_size and img.shape[1] <= max_size):
            raise RuntimeError('The height and width of input images do not meet requirements.')
        image_t = np.empty((max_size, max_size, 3), dtype=img.dtype)
        image_t[:, :] = (104.0, 117.0, 123.0)
        image_t[0:img.shape[0], 0:img.shape[1]] = img
        img = image_t
    return img, resize
