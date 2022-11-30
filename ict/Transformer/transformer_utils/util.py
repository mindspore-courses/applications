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
"""Utils for Transformer."""

import random

import cv2
import numpy as np
import mindspore
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)


def top_k_logits(logits, k):
    """top_k function"""
    topk = mindspore.ops.TopK()
    v, _ = topk(logits, k)
    out = logits
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def sample_mask(model, x, y, length, temperature=1.0, top_k=None):
    """
    Infer utils.

    Args:
        model (mindspore.Model): Model Inference Interface.
        x (Tensor): The input image tensor.
        y (Tensor): The mask image tensor.
        temperature (float): The parameter of softmax. Default: 1.0
        top_k (int): The number of top_k.
    """
    for i in tqdm(range(length)):
        if y[0, i] == 0:
            continue
        logits = model.predict(x, y)
        logits = logits[:, i, :] / temperature
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        softmax = mindspore.ops.Softmax(axis=-1)
        probs = softmax(logits)
        pred = mindspore.ops.multinomial(probs, num_sample=1)
        x[:, i] = pred[:, 0]
        y[:, i] = 0.

    return x


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate_stroke_mask(im_size, max_vertex=25, max_length=100, max_brush_width=24, max_angle=360):
    """Generate random mask function"""
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(5, 13)
    for _ in range(parts):
        mask = mask + np_free_form_mask(max_vertex, max_length, max_brush_width, max_angle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.concatenate([mask, mask, mask], axis=2)
    return mask


def np_free_form_mask(max_vertex, max_length, max_brush_width, max_angle, h, w):
    """Utils of generate random mask function"""
    mask = np.zeros((h, w, 1), np.float32)
    num_vertex = np.random.randint(max_vertex + 1)
    starty = np.random.randint(h)
    startx = np.random.randint(w)
    brush_width = 0
    for i in range(num_vertex):
        angle = np.random.randint(max_angle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(max_length + 1)
        brush_width = np.random.randint(10, max_brush_width + 1) // 2 * 2
        nexty = starty + length * np.cos(angle)
        nextx = startx + length * np.sin(angle)
        nexty = np.maximum(np.minimum(nexty, h - 1), 0).astype(np.int)
        nextx = np.maximum(np.minimum(nextx, w - 1), 0).astype(np.int)
        cv2.line(mask, (starty, startx), (nexty, nextx), 1, brush_width)
        cv2.circle(mask, (starty, startx), brush_width // 2, 2)
        starty, startx = nexty, nextx
    cv2.circle(mask, (starty, startx), brush_width // 2, 2)
    return mask
