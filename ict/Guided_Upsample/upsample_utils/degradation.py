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
"""Utils of generator prior."""

import cv2
import numpy as np
from PIL import Image



def pil_to_np(img_pil):
    '''
    Converts image in PIL format to np.array.
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_pil)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''
    Converts image in np.array format to PIL image.
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def normalize_img(img):
    return img / 127.5 - 1


def squared_euclidean_distance_np(a, b):
    '''
    Calculate the squared Euclidean distance.
    '''
    b = b.T
    a2 = np.sum(np.square(a), axis=1)
    b2 = np.sum(np.square(b), axis=0)
    ab = np.matmul(a, b)
    d = a2[:, None] - 2 * ab + b2[None, :]
    return d


def color_quantize_np(x, clusters):
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance_np(x, clusters)
    return np.argmin(d, axis=1)


def prior_degradation(img, clusters, prior_size):
    '''
    Downsample into 32x32, using origin dictionary cluster to remap the pixel intensity.
    '''
    img_np = np.array(img)

    lr_img_cv2 = cv2.resize(img_np, (prior_size, prior_size), interpolation=cv2.INTER_AREA)
    x_norm = normalize_img(lr_img_cv2)
    token_id = color_quantize_np(x_norm, clusters)
    primers = token_id.reshape(-1, prior_size * prior_size)
    primers_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [prior_size, prior_size, 3]).astype(np.uint8) for s
                   in primers]

    degraded = Image.fromarray(primers_img[0])

    return degraded


def color_quantize_np_topk(x, clusters, k):
    x = x.reshape(-1, 3)
    d = squared_euclidean_distance_np(x, clusters)
    top_k = np.argpartition(d, k, axis=1)[:, :k]
    h, w = top_k.shape
    select_index = np.random.randint(w, size=(h))
    return top_k[range(h), select_index]


def prior_degradation_2(img, clusters, prior_size, k=1):
    '''
    Downsample and random change.
    '''
    lr_img_cv2 = img.resize((prior_size, prior_size), resample=Image.BILINEAR)
    lr_img_cv2 = np.array(lr_img_cv2)
    x_norm = normalize_img(lr_img_cv2)
    token_id = color_quantize_np_topk(x_norm, clusters, k)
    primers = token_id.reshape(-1, prior_size * prior_size)
    primers_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [prior_size, prior_size, 3]).astype(np.uint8) for s
                   in primers]

    degraded = Image.fromarray(primers_img[0])

    return degraded
