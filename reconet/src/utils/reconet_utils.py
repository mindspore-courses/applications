# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.lr_schedule.py
# org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" ReCoNet infer script."""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import Tensor
from mindvision.io import imread
from PIL import Image
import numpy as np


def read_image(image_file):
    """
    Read image as N H W C size

    Args:
        image_file (str): Path of image file.

    Returns:
        Tensor in (N, H, W, C) shape
    """
    image = imread(image_file, 'RGB')
    image = py_vision.ToTensor()(image)
    image = Tensor(image)
    image = ops.ExpandDims()(image, 0)
    return image


def preprocess(image_file):
    """
    Image preprocess for reconet.

    Args:
        image_file (str): Path of image file.

    Returns:
        Tensor, image batch in (N, H, W, C) shape
    """
    image = read_image(image_file)
    return image * 2 - 1


def postprocess(frame):
    """
    Postprocess for image frame to RBG format.

    Args:
        frame (Tensor): Image batch

    Returns:
        ndarray, Image in RBG format
    """
    image = frame.asnumpy()
    input_format = 'CHW'
    index = [input_format.find(c) for c in 'HWC']
    image = image.transpose(index)

    scale_factor = 255
    image = image.astype(np.float32)
    image = (image * scale_factor).astype(np.uint8)
    return image


def save_infer_result(image, file):
    """
    Save infer result to file.

    Args:
        image (Tensor): Image batch
        file (str): Path of saved image file.
    """
    image = postprocess(image)
    image = Image.fromarray(image)
    image.save(file)
    print('save infer result in {}'.format(file))


def batch_style_transfer(batch, model):
    """
    Batch style transfer

    Args:
        batch (Tensor): Image batch
        model (ReCoNet): ReCoNet model

    Returns:
        Tensor, model output batch
    """
    batch = Tensor.from_numpy(np.array(batch)).astype(mindspore.int32)
    batch = ops.Transpose()(batch, (0, 3, 1, 2))
    batch = batch.astype(mindspore.float32) / 255
    batch = batch * 2 - 1
    output = model.predict(batch)
    return (output + 1) / 2


def normalize_batch(batch, mean, std):
    """
    Batch normalize.

    Args:
        batch (Tensor): Image batch
        mean (list): mean of normalize
        std (list): std of normalize

    Returns:
        Tensor, image batch
    """
    dtype = batch.dtype
    mean = Tensor(mean, dtype=dtype)
    std = Tensor(std, dtype=dtype)
    return (batch - mean[None, :, None, None]) / std[None, :, None, None]


def preprocess_for_vgg(images_batch):
    """
    Preprocess for vgg

    Args:
        images_batch (Tensor): Image batch

    Returns:
        Tensor, normalized image batch
    """
    return normalize_batch(images_batch,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])


def gram_matrix(feature_map, device_target):
    """
    Gram vgg feature matrix

    Args:
        feature_map (Tensor): vgg encoded feature
        device_target(str): device type for model to run

    Returns:
        Tensor, vgg encoded feature in (n, c, h, w) shape
    """
    n, c, h, w = feature_map.shape
    cast = ops.Cast()
    if device_target == 'Ascend':
        feature_map = cast(feature_map, mindspore.float16)
    feature_map = feature_map.reshape((n, c, h * w))
    batmatmul = ops.BatchMatMul()
    bmt_result = batmatmul(feature_map, feature_map.transpose(0, 2, 1)) / (c * h * w)
    return cast(bmt_result, mindspore.float32)


def vgg_encode_image(vgg, image, device_target):
    """
    Encoder image by vgg

    Args:
        vgg (Cell): Vgg model
        image (Tensor): Image batch

    Returns:
        List, list of vgg encoded feature
    """
    style = read_image(image)
    style_vgg_features = vgg.encode((preprocess_for_vgg(style)))
    style_gram_matrices = [gram_matrix(x, device_target) for x in style_vgg_features]
    return style_gram_matrices


def magnitude_squared(x):
    """
    Magnitude squared

    Args:
        x (Tensor): Image batch

    Returns:
        float, magnitude squared result
    """
    return ops.Pow()(x, 2).sum(-1)


def nhwc_to_nchw(x):
    """
    Convert tensor shape from (n h w c) to (n c h w)

    Args:
        x (Tensor): Image batch
    Returns:
        Tensor, image batch
    """
    return ops.Transpose()(x, (0, 3, 1, 2))


def nchw_to_nhwc(x):
    """
    Convert tensor shape from (n c h w) to (n h w c)

    Args:
        x (Tensor): Image batch
    Returns:
        Tensor, image batch
    """
    return ops.Transpose()(x, (0, 2, 3, 1))


def custom_grid_sample(im, grid, align_corners=False):
    """
    Grid sample function, modified from mmcv

    Args:
        im (Tensor): Image batch
        grid (Tensor): Grid
        align_corners (bool): Whether align by corners

    Returns:
        Tensor, image batch

        @misc{mmcv,
        title={{MMCV: OpenMMLab} Computer Vision Foundation},
        author={MMCV Contributors},
        howpublished = {url{https://github.com/open-mmlab/mmcv}},
        year={2018}
    }
    """
    n, c, h, w = im.shape
    _, gh, gw, _ = grid.shape

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = ops.Floor()(x).astype('int64')
    y0 = ops.Floor()(y).astype('int64')
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ops.ExpandDims()((x1 - x) * (y1 - y), 1)
    wb = ops.ExpandDims()((x1 - x) * (y - y0), 1)
    wc = ops.ExpandDims()((x - x0) * (y1 - y), 1)
    wd = ops.ExpandDims()((x - x0) * (y - y0), 1)

    # Apply default for grid_sample function zero padding
    im_padded = mindspore.nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")(im)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = mindspore.numpy.where(x0 < 0, Tensor(0), x0)
    x0 = mindspore.numpy.where(x0 > padded_w - 1, Tensor(0), x0)
    x1 = mindspore.numpy.where(x1 < 0, Tensor(0), x1)
    x1 = mindspore.numpy.where(x1 > padded_w - 1, Tensor(0), x1)
    y0 = mindspore.numpy.where(y0 < 0, Tensor(0), y0)
    y0 = mindspore.numpy.where(y0 > padded_h - 1, Tensor(0), y0)
    y1 = mindspore.numpy.where(y1 < 0, Tensor(0), y1)
    y1 = mindspore.numpy.where(y1 > padded_h - 1, Tensor(0), y1)

    im_padded = im_padded.view(n, c, -1)

    broadcast = ops.BroadcastTo((-1, c, -1))
    x0_y0 = ops.ExpandDims()(x0 + y0 * padded_w, 1)
    x0_y0 = broadcast(x0_y0)
    x0_y1 = ops.ExpandDims()(x0 + y1 * padded_w, 1)
    x0_y1 = broadcast(x0_y1)
    x1_y0 = ops.ExpandDims()(x1 + y0 * padded_w, 1)
    x1_y0 = broadcast(x1_y0)
    x1_y1 = ops.ExpandDims()(x1 + y1 * padded_w, 1)
    x1_y1 = broadcast(x1_y1)

    padded_a = ops.GatherD()(im_padded, 2, x0_y0)
    padded_b = ops.GatherD()(im_padded, 2, x0_y1)
    padded_c = ops.GatherD()(im_padded, 2, x1_y0)
    padded_d = ops.GatherD()(im_padded, 2, x1_y1)

    return (padded_a * wa + padded_b * wb + padded_c * wc + padded_d * wd).reshape(n, c, gh, gw)


def warp_optical_flow(source, reverse_flow):
    """
    Warp optical flow

    Args:
        source (Tensor): Source optical flow
        reverse_flow (Tensor): Reverse optical flow

    Returns:
        Tensor, wrapped optical flow batch
    """
    _, h, w, _ = reverse_flow.shape

    reverse_flow = reverse_flow.copy()
    reverse_flow[..., 0] += mindspore.numpy.arange(w).view(1, 1, w)
    reverse_flow[..., 0] *= 2 / w
    reverse_flow[..., 0] -= 1
    reverse_flow[..., 1] += mindspore.numpy.arange(h).view(1, h, 1)
    reverse_flow[..., 1] *= 2 / h
    reverse_flow[..., 1] -= 1
    value = custom_grid_sample(source, reverse_flow)
    return value


def occlusion_mask_from_flow(optical_flow, reverse_optical_flow, motion_boundaries):
    """
    Get occlusion mask from optical flow

    Args:
        optical_flow (Tensor): Optical flow
        reverse_optical_flow (Tensor): Reverse optical flow
        motion_boundaries (Tensor): Motion boundaries

    Returns:
        Tensor, occlusion_mask batch
    """
    optical_flow = nhwc_to_nchw(optical_flow)
    optical_flow = warp_optical_flow(optical_flow, reverse_optical_flow)
    optical_flow = nchw_to_nhwc(optical_flow)

    forward_magnitude = magnitude_squared(optical_flow)
    reverse_magnitude = magnitude_squared(reverse_optical_flow)
    sum_magnitude = magnitude_squared(optical_flow + reverse_optical_flow)

    occlusion_mask = sum_magnitude < (0.01 * (forward_magnitude + reverse_magnitude) + 0.5)
    occlusion_mask = Tensor(occlusion_mask.asnumpy() & ~motion_boundaries.asnumpy())
    return ops.ExpandDims()(occlusion_mask.astype('float32'), 1)


def resize_optical_flow(optical_flow, h, w):
    """
    Resize optical flow

    Args:
        optical_flow (Tensor): Optical flow
        h (int): Height of optical flow
        w (int): Weight of optical flow

    Returns:
        Tensor, optical flow
    """
    optical_flow_nchw = nhwc_to_nchw(optical_flow)
    optical_flow_resized_nchw = nn.ResizeBilinear()(optical_flow_nchw, (h, w))
    optical_flow_resized = nchw_to_nhwc(optical_flow_resized_nchw)

    old_h, old_w = optical_flow_nchw.shape[-2:]
    h_scale, w_scale = h / old_h, w / old_w
    optical_flow_resized[..., 0] *= w_scale
    optical_flow_resized[..., 1] *= h_scale
    return optical_flow_resized


def rgb_to_luminance(x):
    """
    RBG to luminance

    Args:
        x (Tensor): Image batch

    Returns:
        Tensor, luminance batch
    """
    return x[:, 0, ...] * 0.2126 + x[:, 1, ...] * 0.7512 + x[:, 2, ...] * 0.0722
