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
"""
An unconventional data enhancement method, a simple data enhancement principle independent of data.
"""
import numpy as np

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops as P


def one_hot(x, num_classes, on_value=1., off_value=0.):
    """
    Coding the label for unique heat.

    Args:
        x (Tensor): Parameter.
        num_classes (int): Number of class.
        on_value (float): Max value.
        off_value (float): Min value.

    Returns:
        Array, replace x that is less than off_value with off_value and greater than on_value with on_value.
    """
    x = x.reshape(-1)

    # The np.eye is used to generate a two-dimensional array of num_classes*num_classes
    # with the elements on the diagonal being 1 and the rest being 0
    x = np.eye(num_classes)[x]

    # Replace x that is less than off_value with off_value and greater than on_value with on_value
    x = np.clip(x, a_min=off_value, a_max=on_value, dtype=np.float32)
    return x


def mix_up_target(target, num_classes, lam=1., smoothing=0.0):
    """
    MixUp the target.

    Args:
        target (Tensor): Target.
        num_classes (int): Number of class.
        lam (float): Coefficient.
        smoothing (float): Smoothing factor.

    Returns:
        Float: MixUp the target.
    """
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value)
    y2 = one_hot(np.flip(target, axis=0), num_classes, on_value=on_value, off_value=off_value)
    return y1 * lam + y2 * (1. - lam)


def rand_bbox(img_shape, lam, margin=0., count=None):
    """
    Generate random borders.

    Args:
        img_shape (int): Image shape.
        lam (float): Coefficient.
        margin (float): Margin.
        count (int): Size count.

    Returns:
        Tensor, rand bbox.
    """

    # margin is the percentage of the bbox dimension to be forced as a margin
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """
    Mixed bounding box.

    Args:
        img_shape (int): Image shape.
        minmax (tuple): Min and max.
        count (int): Size count.

    Returns:
        Tensor, rand bbox with min and max.
    """

    # count is the number of bboxes to be generated
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cut_mix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    """
    Generate bbox and perform lambda correction

    Args:
        img_shape (int): Image shape.
        lam (float): Coefficient.
        ratio_minmax (float): Min and max ratio.
        correct_lam (bool): whether correct lam.
        count (int): Size count.

    Returns:
        Tensor, bbox and lam.
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class MixUp:
    """
    MixUp and CutMix with different parameters for each element or the whole batch.

    Args:
        mix_up_alpha (float): MixUp alpha.
        cut_mix_alpha (float): Cut mix alpha.
        cut_mix minmax (tuple): Min and max ratio when cut mix.
        prob (float): Probability of mix_up or cut_mix for each batch or element
        switch_prob (float): Probability of switching to cut_mix instead of mix_up
        mode (str): Mode.
        correct_lam (bool): Apply lambda correction when cut_mix bbox is clipped by image border.
        label_smoothing (float): Apply label smoothing to the blended target tensor
        num_classes (int): Number of class.
    """

    def __init__(self,
                 mix_up_alpha=1.,
                 cut_mix_alpha=0.,
                 cut_mix_minmax=None,
                 prob=1.0,
                 switch_prob=0.5,
                 mode='batch',
                 correct_lam=True,
                 label_smoothing=0.1,
                 num_classes=1000):
        self.mix_up_alpha = mix_up_alpha
        self.cut_mix_alpha = cut_mix_alpha
        self.cut_mix_minmax = cut_mix_minmax
        if self.cut_mix_minmax is not None:
            assert len(self.cut_mix_minmax) == 2
            self.cut_mix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam
        self.mix_up_enabled = True

    def _params_per_elem(self, batch_size):
        """Parameters for each element."""
        lam = np.ones(batch_size, dtype=np.float32)
        use_cut_mix = np.zeros(batch_size, dtype=np.bool)
        if self.mix_up_enabled:
            if self.mix_up_alpha > 0. and self.cut_mix_alpha > 0.:
                use_cut_mix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cut_mix,
                    np.random.beta(self.cut_mix_alpha, self.cut_mix_alpha, size=batch_size),
                    np.random.beta(self.mix_up_alpha, self.mix_up_alpha, size=batch_size))
            elif self.mix_up_alpha > 0.:
                lam_mix = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, size=batch_size)
            elif self.cut_mix_alpha > 0.:
                use_cut_mix = np.ones(batch_size, dtype=np.bool)
                lam_mix = np.random.beta(self.cut_mix_alpha, self.cut_mix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cut_mix

    def _params_per_batch(self):
        """Parameters for each batch."""
        lam = 1.
        use_cut_mix = False
        if self.mix_up_enabled and np.random.rand() < self.mix_prob:
            if self.mix_up_alpha > 0. and self.cut_mix_alpha > 0.:
                use_cut_mix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cut_mix_alpha, self.cut_mix_alpha) if use_cut_mix else \
                    np.random.beta(self.mix_up_alpha, self.mix_up_alpha)
            elif self.mix_up_alpha > 0.:
                lam_mix = np.random.beta(self.mix_up_alpha, self.mix_up_alpha)
            elif self.cut_mix_alpha > 0.:
                use_cut_mix = True
                lam_mix = np.random.beta(self.cut_mix_alpha, self.cut_mix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cut_mix

    def _mix_elem(self, x):
        """Blend the elements."""
        batch_size = len(x)
        lam_batch, use_cut_mix = self._params_per_elem(batch_size)
        x_orig = x.clone()
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cut_mix[i]:
                    (yl, yh, xl, xh), lam = cut_mix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cut_mix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return P.ExpandDims()(Tensor(lam_batch, dtype=mstype.float32), 1)

    def _mix_pair(self, x):
        """Mixing pair for pair."""
        batch_size = len(x)
        lam_batch, use_cut_mix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cut_mix[i]:
                    (yl, yh, xl, xh), lam = cut_mix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cut_mix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return P.ExpandDims()(Tensor(lam_batch, dtype=mstype.float32), 1)

    def _mix_batch(self, x):
        """Mixing the batches."""
        lam, use_cut_mix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cut_mix:
            (yl, yh, xl, xh), lam = cut_mix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cut_mix_minmax, correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = np.flip(x, axis=0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = np.flip(x, axis=0) * (1. - lam)
            x *= lam
            x += x_flipped
        return lam

    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mix_up_target(target, self.num_classes, lam, self.label_smoothing)
        return x.astype(np.float32), target.astype(np.float32)
