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
"""Define Head"""

import numpy as np
import cv2

import mindspore
from mindspore.common.initializer import initializer, Normal
import mindspore.nn as nn
import mindspore.ops as ops


class ViPNASHeatmapSimpleHead(nn.Cell):
    """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        num_deconv_groups (list|tuple): Group number.
        in_index (int|Sequence[int]): Input feature index. Default: -1
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_deconv_layers=3,
                 num_deconv_filters=(144, 144, 144),
                 num_deconv_kernels=(4, 4, 4),
                 num_deconv_groups=(16, 16, 16),
                 in_index=0,
                 align_corners=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.num_deconv_layers = num_deconv_layers

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self._init_inputs(in_channels, in_index)
        self.in_index = in_index
        self.align_corners = align_corners

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers, num_deconv_filters,
                num_deconv_kernels, num_deconv_groups)
        elif num_deconv_layers < 0:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        kernel_size = 1
        padding = 0

        conv_channels = num_deconv_filters[-1] if num_deconv_layers > 0 else self.in_channels

        self.final_layer = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            pad_mode='pad',
            padding=padding,
            has_bias=True,
            weight_init=Normal(sigma=0.001),
            bias_init='zeros')

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (mindspore.Tensor[NxKxHxW]): Output heatmaps.
            target (mindspore.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (mindspore.Tensor[NxKx1]):
                Weights across different joint types.
        """

        losses = dict()
        if target.dim() != 4 or target_weight.dim() != 3:
            raise KeyError('target.dim or target_weight.dim() get wrong value')

        criterion = nn.MSELoss()

        batch_size, num_joints, _, _ = output.shape

        split = ops.Split(1, num_joints)
        heatmaps_pred = split(output.reshape((batch_size, num_joints, -1)))
        heatmaps_gt = split(target.reshape((batch_size, num_joints, -1)))

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            loss += criterion(heatmap_pred * target_weight[:, idx],
                              heatmap_gt * target_weight[:, idx])

        losses['mse_loss'] = loss / num_joints

        return losses['mse_loss']

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (np.ndarray[NxKxHxW]): Output heatmaps.
            target (mindspore.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (mindspore.Tensor[NxKx1]):
                Weights across different joint types.
        """

        accuracy = dict()

        _, avg_acc, _ = pose_pck_accuracy(
            output.copy().asnumpy(),
            target.copy().asnumpy(),
            target_weight.copy().asnumpy().squeeze(-1) > 0)
        accuracy['acc_pose'] = float(avg_acc)

        return accuracy['acc_pose']

    def construct(self, x):
        """Construct function."""
        x = self._transform_inputs(x)
        if self.num_deconv_layers > 0:
            x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (mindspore.Tensor[NxKxHxW]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.construct(x)

        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.copy().asnumpy(),
                flip_pairs)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.copy().asnumpy()
        return output_heatmap

    def decode(self, img_metas, output):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_heatmaps(output, c, s)

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    def _init_inputs(self, in_channels, in_index):
        """Check and initialize input transforms.

        The in_channels and in_index must match.
        Only single feature map will be selected. So in_channels and in_index must be of type int.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
        """

        self.in_index = in_index
        if not isinstance(in_channels, int):
            raise TypeError('in_channels should be int')
        if not isinstance(in_index, int):
            raise TypeError('in_index should be int')
        self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs

        inputs = inputs[self.in_index]

        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels,
                           num_groups):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)
        if num_layers != len(num_groups):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_groups({len(num_groups)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            if kernel == 4:
                padding = 1
            elif kernel == 3:
                padding = 1
            elif kernel == 2:
                padding = 0
            else:
                raise ValueError(f'Not supported num_kernels ({kernel}).')

            planes = num_filters[i]
            groups = num_groups[i]
            layers.append(
                nn.Conv2dTranspose(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    group=groups,
                    padding=padding,
                    weight_init=Normal(sigma=0.001),
                    has_bias=False,
                    pad_mode='pad')
            )
            layers.append(nn.BatchNorm2d(num_features=planes))
            layers.append(nn.ReLU())
            self.in_channels = planes

        return nn.SequentialCell(*layers)

def normal_init(module, mean=0, std=1., bias=0):
    if hasattr(module, 'weight'):
        module.weight = initializer(Normal(sigma=std, mean=mean), module.weight.shape, mindspore.float32)
    if hasattr(module, 'bias'):
        module.bias = initializer(bias, module.bias.shape, mindspore.float32)
    if hasattr(module, 'gamma_init'):
        module.gamma_init = initializer(Normal(sigma=std, mean=mean), module.gamma_init.shape, mindspore.float32)
    if hasattr(module, 'beta_init'):
        module.beta_init = initializer(bias, module.beta_init.shape, mindspore.float32)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight'):
        module.weight = initializer(val, module.weight.shape, mindspore.float32)
    if hasattr(module, 'bias'):
        module.bias = initializer(bias, module.bias.shape, mindspore.float32)
    if hasattr(module, 'gamma'):
        module.gamma = initializer(val, module.gamma.shape, mindspore.float32)
    if hasattr(module, 'beta'):
        module.beta = initializer(bias, module.beta.shape, mindspore.float32)


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.
        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    if not isinstance(heatmaps, np.ndarray):
        raise TypeError('heatmaps should be numpy.ndarray')
    if heatmaps.ndim != 4:
        raise KeyError('batch_images should be 4-ndim')

    n, k, _, w = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((n, k, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((n, k, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((n, k, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % w
    preds[:, :, 1] = preds[:, :, 1] // w

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _calc_distances(preds, targets, mask, normalize):
    """Calculate the normalized distances between preds and target.

    Note:
        batch_size: N
        num_keypoints: K
        dimension of keypoints: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        targets (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (np.ndarray[N, D]): Typical value is heatmap_size

    Returns:
        np.ndarray[K, N]: The normalized distances.
          If target keypoints are missing, the distance is -1.
    """
    n, k, _ = preds.shape
    distances = np.full((n, k), -1, dtype=np.float32)
    # handle invalid values
    normalize[np.where(normalize <= 0)] = 1e6
    distances[mask] = np.linalg.norm(((preds - targets) / normalize[:, None, :])[mask], axis=-1)
    return distances.T


def _distance_acc(distances):
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        batch_size: N

    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold.
          If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < 0.5).sum() / num_distance_valid
    return -1


def keypoint_pck_accuracy(pred, gt, mask, normalize):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.
        batch_size: N
        num_keypoints: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.
        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, normalize)

    acc = np.array([_distance_acc(d) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0
    return acc, avg_acc, cnt


def pose_pck_accuracy(output, target, mask, normalize=None):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints from heatmaps.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        output (np.ndarray[N, K, H, W]): Model output heatmaps.
        target (np.ndarray[N, K, H, W]): Groundtruth heatmaps.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation. Default 0.05.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.
        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    n, k, h, w = output.shape
    if k == 0:
        return None, 0, 0
    if normalize is None:
        normalize = np.tile(np.array([[h, w]]), (n, 1))

    pred, _ = _get_max_preds(output)
    gt, _ = _get_max_preds(target)
    return keypoint_pck_accuracy(pred, gt, mask, normalize)


def flip_back(output_flipped, flip_pairs):
    """Flip the flipped heatmaps back to the original form.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        output_flipped (np.ndarray[N, K, H, W]): The output heatmaps obtained
            from the flipped images.
        flip_pairs (list[tuple()): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).

    Returns:
        np.ndarray: heatmaps that flipped back to the original image
    """
    if output_flipped.ndim != 4:
        raise KeyError('output_flipped should be [batch_size, num_keypoints, height, width]')
    shape_ori = output_flipped.shape
    channels = 1
    output_flipped = output_flipped.reshape(shape_ori[0], -1, channels,
                                            shape_ori[2], shape_ori[3])
    output_flipped_back = output_flipped.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
    output_flipped_back = output_flipped_back.reshape(shape_ori)
    # Flip horizontally
    output_flipped_back = output_flipped_back[..., ::-1]
    return output_flipped_back


def keypoints_from_heatmaps(heatmaps, center, scale):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        batch size: N
        num keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """

    h_n, h_k, h_h, h_w = heatmaps.shape
    preds, maxvals = _get_max_preds(heatmaps)
    # add +/-0.25 shift to the predicted locations for higher acc.
    for n in range(h_n):
        for k in range(h_k):
            heatmap = heatmaps[n][k]
            px = int(preds[n][k][0])
            py = int(preds[n][k][1])
            if 1 < px < W - 1 and 1 < py < h_h - 1:
                diff = np.array([
                    heatmap[py][px + 1] - heatmap[py][px - 1],
                    heatmap[py + 1][px] - heatmap[py - 1][px]
                ])
                preds[n][k] += np.sign(diff) * .25

    # Transform back to the image
    for i in range(h_n):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [h_w, h_h])

    return preds, maxvals


def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray[N, K, H, W]: Modulated heatmap distribution.
    """
    if kernel % 2 != 1:
        raise KeyError('kernel % 2 should equal to 1')

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        batch size: B
        num keypoints: K
        num persons: N
        height of heatmaps: H
        width of heatmaps: W
        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        res (np.ndarray[N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    b, k, h, w = batch_heatmaps.shape
    n = coords.shape[0]
    if b not in (1, n):
        raise ValueError('B should equal to 1 or N')
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)
    batch_heatmaps = np.transpose(batch_heatmaps,
                                  (2, 3, 0, 1)).reshape(h, w, -1)
    batch_heatmaps_pad = cv2.copyMakeBorder(
        batch_heatmaps, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
    batch_heatmaps_pad = np.transpose(
        batch_heatmaps_pad.reshape(H + 2, W + 2, b, K),
        (2, 3, 0, 1)).flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (w + 2)
    index += (w + 2) * (h + 2) * np.arange(0, b * k).reshape(-1, k)
    index = index.astype(np.int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + w + 2]
    ix1y1 = batch_heatmaps_pad[index + w + 3]
    ix1_y1_ = batch_heatmaps_pad[index - w - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - w]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape([n, k, 2, 1])
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape([n, k, 2, 2])
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        heatmap height: H
        heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    h, w = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < w - 2 and 1 < py < h - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
            heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
            heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
            heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def transform_preds(coords, center, scale, output_size):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    if coords.shape[1] not in (2, 4, 5):
        raise KeyError('coords.shape[1] should in (2, 4, 5)')
    if len(center) != 2:
        raise KeyError('The length of center should be 2')
    if len(scale) != 2:
        raise KeyError('The length of scale should be 2')
    if len(output_size) != 2:
        raise KeyError('The length of output_size should be 2')

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    scale_x = scale[0] / output_size[0]
    scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords
