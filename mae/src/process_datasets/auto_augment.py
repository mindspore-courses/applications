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
# =========================================================================
"""
Search the optimal data enhancement strategy through reinforcement learning violence.
"""

import math
import random
import re

import numpy as np
import PIL
from PIL import Image, ImageOps, ImageEnhance

_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (128, 128, 128)

# This signifies the max integer that the controller RNN could predict for the augmentation scheme.
_MAX_LEVEL = 10.

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,
)

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _interpolation(kwargs):
    """
    Infer function values at other points from function values at some known points.

    Args:
        kwargs (Tensor): Pass an indefinite number of parameters to a function.

    Returns:
        Str, selected interpolation method.
    """
    interpolation = kwargs.pop('resample', Image.BILINEAR)
    interpolation = random.choice(interpolation) if isinstance(interpolation, (list, tuple)) else interpolation
    return interpolation


def _check_args_tf(kwargs):
    """
    Determine whether to fill color.

    Args:
        kwargs (Tensor): Pass an indefinite number of parameters to a function.
    """
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    """
    Share the image along the horizontal axis with rate magnitude.

    Args:
        img (Image): Image to be processed.
        factor (float): Range of magnitude.
        **kwargs (Tensor): Pass an indefinite number of parameters to a function.

    Returns:
        Image, processed image.
    """
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    """
    Share the image along the vertical axis with rate magnitude.

    Args:
        img (Image): Image to be processed.
        factor (float): Range of magnitude.
        **kwargs (Tensor): Pass an indefinite number of parameters to a function.

    Returns:
        Image, processed image.
    """
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    """
    Translate the image in the horizontal direction by magnitude number of pixels (relative displacement of nodes).

    Args:
        img (Image): Image to be processed.
        pct (float): Range of magnitude.
        **kwargs (Tensor): Pass an indefinite number of parameters to a function.

    Returns:
        Image, processed image.
    """
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    """
    Translate the image in the vertical direction by magnitude number of pixels (relative displacement of nodes).

    Args:
        img (Image): Image to be processed.
        pct (float): Range of magnitude.
        **kwargs (Tensor): Pass an indefinite number of parameters to a function.

    Returns:
        Image, processed image.
    """
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    """
    Translate the image in the horizontal direction by magnitude number of pixels (Absolute displacement of nodes).

    Args:
        img (Image): Image to be processed.
        pixels (float): Range of magnitude.
        **kwargs (Tensor): Pass an indefinite number of parameters to a function.

    Returns:
        Image, processed image.
    """
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    """
    Translate the image in the vertical direction by magnitude number of pixels (Absolute displacement of nodes).

    Args:
        img (Image): Image to be processed.
        pixels (float): Range of magnitude.
        **kwargs (Tensor): Pass an indefinite number of parameters to a function.

    Returns:
        Image, processed image.
    """
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    """
    Rotate the image magnitude degrees.

    Args:
        img (Image): Image to be processed.
        degrees (float): Range of rotate.
        **kwargs (Tensor): Pass an indefinite number of parameters to a function.

    Returns:
        Image, processed image.
    """
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        func = img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        func = img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        func = img.rotate(degrees, resample=kwargs['resample'])
    return func


def auto_contrast(img, **__):
    """
    Maximize the image contrast, by making the darkest pixel black and lightest pixel white.

    Args:
        img (Image): Image to be processed.

    Returns:
        Image, processed image.
    """
    return ImageOps.autocontrast(img)


def invert(img, **__):
    """
    Invert the pixels of the image.

    Args:
        img (Image): Image to be processed.

    Returns:
        Image, processed image.
    """
    return ImageOps.invert(img)


def equalize(img, **__):
    """
    Equalize the image histogram.

    Args:
        img (Image): Image to be processed.

    Returns:
        Image, processed image.
    """
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    """
    Invert all pixels above a threshold value of magnitude.

    Args:
        img (Image): Image to be processed.
        thresh (float): range of magnitudes.

    Returns:
        Image, processed image.
    """
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    """
    Invert all pixels above a threshold value of magnitude.

    Args:
        img (Image): Image to be processed.
        add (float): Less than thresh requires additional values.
        thresh (float): range of magnitudes.

    Returns:
        Image, processed image.
    """
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        func = img.point(lut)
    else:
        func = img
    return func


def posterize(img, bits_to_keep, **__):
    """
    Reduce the number of bits for each pixel to magnitude bits.

    Args:
        img (Image): Image to be processed.
        bits_to_keep (int): range of magnitudes.

    Returns:
        Image, processed image.
    """
    if bits_to_keep >= 8:
        func = img
    else:
        func = ImageOps.posterize(img, bits_to_keep)
    return func


def contrast(img, factor, **__):
    """
    Control the contrast of the image.
    A factor=0 gives a gray image, whereas factor=1 gives the original image.

    Args:
        img (Image): Image to be processed.
        factor (float): range of magnitudes.

    Returns:
        Image, processed image.
    """
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    """
    Adjust the color balance of the image, in a manner similar to the controls on a colour TV set.
    A factor=0 gives a black & white image, whereas factor=1 gives the original image.

    Args:
        img (Image): Image to be processed.
        factor (float): range of magnitudes.

    Returns:
        Image, processed image.
    """
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    """
    Adjust the brightness of the image.
    A factor=0 gives a black image, whereas factor=1 gives the original image.

    Args:
        img (Image): Image to be processed.
        factor (float): range of magnitudes.

    Returns:
        Image, processed image.
    """
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    """
    Adjust the sharpness of the image.
    A factor=0 gives a blurred image, whereas factor=1 gives the original image.

    Args:
        img (Image): Image to be processed.
        factor (float): range of magnitudes.

    Returns:
        Image, processed image.
    """
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """
    With 50% prob, negate the value.

    Args:
        v (float): Value when not negate.

    Returns:
        Bool, whether random negate is performed.
    """
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, hparams):
    """
    The level of randomly negate.

    Args:
        level (float): Range [-30, 30].

    Returns:
        Float, level value.
    """
    translate_ckpt = hparams.get('translate_ckpt', 30.)
    level = (level / _MAX_LEVEL) * translate_ckpt
    level = _randomly_negate(level)
    return (level,)


def _enhance_level_to_arg(level, hparams):
    """
        Calculate the degree of enhancement.

        Args:
            level (float): Range [0.1, 1.9].

        Returns:
            Float, enhance level value.
        """
    translate_ckpt = hparams.get('translate_ckpt', 1.8)
    return ((level / _MAX_LEVEL) * translate_ckpt + 0.1, hparams)


def _enhance_increasing_level_to_arg(level, hparams):
    """
    Calculate the degree of enhance increasing.

    Args:
        level (float): Range [0.1, 1.9], the 'no change' level is 1.0.

    Returns:
        Float, enhance increasing level value.
    """
    translate_ckpt = hparams.get('translate_ckpt', 1.0)
    level = (level / _MAX_LEVEL) * .9
    level = translate_ckpt + _randomly_negate(level)
    return (level,)


def _shear_level_to_arg(level, hparams):
    """
    Calculate the degree of shear.

    Args:
        level (float): Range [-0.3, 0.3].

    Returns:
        Float, shear level value.
    """
    translate_ckpt = hparams.get('translate_ckpt', 0.3)
    level = (level / _MAX_LEVEL) * translate_ckpt
    level = _randomly_negate(level)
    return (level,)


def _translate_abs_level_to_arg(level, hparams):
    """
    Calculate the degree of absolute displacement of nodes.

    Args:
        level (float): Default range [-0.45, 0.45].
        hparams (dict): Parameters.

    Returns:
        Float, absolute displacement of nodes value.
    """
    translate_const = hparams['translate_const']
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return (level,)


def _translate_rel_level_to_arg(level, hparams):
    """
    Calculate the degree of relative displacement of nodes.

    Args:
        level (float): Default range [-0.45, 0.45].
        hparams (dict): Parameters.

    Returns:
        Float, relative displacement of nodes value.
    """
    translate_pct = hparams.get('translate_pct', 0.45)
    level = (level / _MAX_LEVEL) * translate_pct
    level = _randomly_negate(level)
    return (level,)


def _posterize_level_to_arg(level, hparams):
    """
    Calculate the degree of the number of bits for each pixel to magnitude bits.
    Intensity or Severity of augmentation decreases with level.

    Args:
        level (float): Range [0, 4].

    Returns:
        int, the number of bits.
    """
    translate_pct = hparams.get('translate_pct', 4)
    return (int((level / _MAX_LEVEL) * translate_pct),)


def _posterize_increasing_level_to_arg(level, hparams):
    """
    Calculate the increasing degree of the number of bits for each pixel to magnitude bits.
    Intensity or Severity of augmentation decreases with level.

    Args:
        level (float): Range [4, 0].

    Returns:
        int, the number of bits.
    """
    return (4 - _posterize_level_to_arg(level, hparams)[0],)


def _posterize_original_level_to_arg(level, hparams):
    """
    Calculate the original degree of the number of bits for each pixel to magnitude bits.
    Intensity or Severity of augmentation decreases with level.

    Args:
        level (float): Range [4, 0].

    Returns:
        int, the number of bits.
    """
    translate_ckpt = hparams.get('translate_ckpt', 4)
    return (int((level / _MAX_LEVEL) * translate_ckpt) + 4,)


def _solarize_level_to_arg(level, hparams):
    """
    Calculate the degree of invert all pixels above a threshold value of magnitude.
    Intensity or Severity of augmentation decreases with level.

    Args:
        level (float): Range [0, 256].

    Returns:
        int, the degree of solarize.
    """
    translate_ckpt = hparams.get('translate_ckpt', 256)
    return (int((level / _MAX_LEVEL) * translate_ckpt),)


def _solarize_increasing_level_to_arg(level, hparams):
    """
        Calculate the increasing degree of invert all pixels above a threshold value of magnitude.
        Intensity or Severity of augmentation decreases with level.

        Args:
            level (float): Range [0, 256].

        Returns:
            int, the degree of solarize increasing.
        """
    return (256 - _solarize_level_to_arg(level, hparams)[0],)


def _solarize_add_level_to_arg(level, hparams):
    """
        Calculate the add degree of invert all pixels above a threshold value of magnitude.
        Intensity or Severity of augmentation decreases with level.

        Args:
            level (float): Range [0, 110].

        Returns:
            int, the degree of solarize add.
        """
    translate_ckpt = hparams.get('translate_ckpt', 110)
    return (int((level / _MAX_LEVEL) * translate_ckpt),)


LEVEL_TO_ARG = {
    'AutoContrast': None,
    'Equalize': None,
    'Invert': None,
    'Rotate': _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    'Posterize': _posterize_level_to_arg,
    'PosterizeIncreasing': _posterize_increasing_level_to_arg,
    'PosterizeOriginal': _posterize_original_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeIncreasing': _solarize_increasing_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'ColorIncreasing': _enhance_increasing_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'ContrastIncreasing': _enhance_increasing_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'BrightnessIncreasing': _enhance_increasing_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'SharpnessIncreasing': _enhance_increasing_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_abs_level_to_arg,
    'TranslateY': _translate_abs_level_to_arg,
    'TranslateXRel': _translate_rel_level_to_arg,
    'TranslateYRel': _translate_rel_level_to_arg,
}

NAME_TO_OP = {
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'PosterizeIncreasing': posterize,
    'PosterizeOriginal': posterize,
    'Solarize': solarize,
    'SolarizeIncreasing': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'ColorIncreasing': color,
    'Contrast': contrast,
    'ContrastIncreasing': contrast,
    'Brightness': brightness,
    'BrightnessIncreasing': brightness,
    'Sharpness': sharpness,
    'SharpnessIncreasing': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
}


class AugmentOp:
    """
        Encapsulate the relevant operations required for automatic enhancement.

        Args:
            name (str): Name of the operation.
            prob (float): Probability value.
            magnitude (float): The probability of this operation and the amplitude of image enhancement.
            hparams (dict): Parameters.
        """

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
        )

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        # If magnitude_std is inf, we sample magnitude from a uniform distribution
        self.magnitude_std = self.hparams.get('magnitude_std', 0)

    def __call__(self, img):
        """
        Apply augment.

        Args:
            img (Image): Image to be processed.

        Returns:
            Image, processed image.
        """
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std:
            if self.magnitude_std == float('inf'):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude))  # clip to valid range
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)


def auto_augment_policy_v0(hparams):
    """
    ImageNet v0 policy from TPU EfficientNet implement.

    Args:
        hparams (dict): Parameters.

    Returns:
        Class, encapsulated automatic enhancement.
    """
    policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateYRel', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],  # This results in black image with Tpu posterize
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_v0r(hparams):
    """
    ImageNet v0 policy from TPU EfficientNet implement, with variation of Posterize used.
    In Google research implementation (number of bits discarded increases with magnitude).

    Args:
        hparams (dict): Parameters.

    Returns:
        Class, encapsulated automatic enhancement.
    """
    policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('PosterizeIncreasing', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateYRel', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('PosterizeIncreasing', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_original(hparams):
    """
    The original policy of auto augment.

    Args:
        hparams (dict): Parameters.

    Returns:
        Class, encapsulated automatic enhancement.
    """
    policy = [
        [('PosterizeOriginal', 0.4, 8), ('Rotate', 0.6, 9)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        [('PosterizeOriginal', 0.6, 7), ('PosterizeOriginal', 0.6, 6)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
        [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
        [('PosterizeOriginal', 0.8, 5), ('Equalize', 1.0, 2)],
        [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
        [('Equalize', 0.6, 8), ('PosterizeOriginal', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
        [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
        [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
        [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_originalr(hparams):
    """
    The originalr policy of auto augment, with research posterize variation.

    Args:
        hparams (dict): Parameters.

    Returns:
        Class, encapsulated automatic enhancement.
    """
    policy = [
        [('PosterizeIncreasing', 0.4, 8), ('Rotate', 0.6, 9)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        [('PosterizeIncreasing', 0.6, 7), ('PosterizeIncreasing', 0.6, 6)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
        [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
        [('PosterizeIncreasing', 0.8, 5), ('Equalize', 1.0, 2)],
        [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
        [('Equalize', 0.6, 8), ('PosterizeIncreasing', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
        [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
        [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
        [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy(name='v0', hparams=None):
    """
    Select the specified enhancement policy according to the entered string.

    Args:
        name (str): specified enhancement policy.

    Returns:
        Class, encapsulated automatic enhancement.
    """
    hparams = hparams or _HPARAMS_DEFAULT
    if name == 'original':
        func = auto_augment_policy_original(hparams)
    elif name == 'originalr':
        func = auto_augment_policy_originalr(hparams)
    elif name == 'v0':
        func = auto_augment_policy_v0(hparams)
    elif name == 'v0r':
        func = auto_augment_policy_v0r(hparams)
    else:
        assert False, 'Unknown AA policy (%s)' % name
    return func


class AutoAugment:
    """
    Auto enhance policy encapsulation class.

    Args:
        policy (list): A collection of enhancement strategies that may be used.
    """
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, img):
        """
        Apply auto augment.

        Args:
            img (Image): Image to be processed.

        Returns:
            Image, processed image.
        """
        sub_policy = random.choice(self.policy)
        for op in sub_policy:
            img = op(img)
        return img


def auto_augment_transform(config_str, hparams):
    """
    Create a AutoAugment transform.

    Args:
        config_str (str): string defining configuration of auto augmentation.
        hparams (dict): other kwargs for the AutoAugmentation scheme.

    Returns:
        Class, a MindSpore compatible transform.
    """
    config = config_str.split('-')
    policy_name = config[0]
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param injected via hparams for now.
            hparams.setdefault('magnitude_std', float(val))
        else:
            assert False, 'Unknown AutoAugment config section'
    aa_policy = auto_augment_policy(policy_name, hparams=hparams)
    return AutoAugment(aa_policy)


_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
]

_RAND_INCREASING_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'PosterizeIncreasing',
    'SolarizeIncreasing',
    'SolarizeAdd',
    'ColorIncreasing',
    'ContrastIncreasing',
    'BrightnessIncreasing',
    'SharpnessIncreasing',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
]

# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_CHOICE_WEIGHTS_0 = {
    'Rotate': 0.3,
    'ShearX': 0.2,
    'ShearY': 0.2,
    'TranslateXRel': 0.1,
    'TranslateYRel': 0.1,
    'Color': .025,
    'Sharpness': 0.025,
    'AutoContrast': 0.025,
    'Solarize': .005,
    'SolarizeAdd': .005,
    'Contrast': .005,
    'Brightness': .005,
    'Equalize': .005,
    'Posterize': 0,
    'Invert': 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    """
    Select a random weight value.

    Args:
        weight_idx (int): Only one set of weights currently.
        transforms (dict): Strategies for transforming and enriching images.

    Returns:
        Float, calculated probability value.
    """
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    """
    Select the enhancement action to adopt.

    Args:
        magnitude (float): The probability of this operation and the amplitude of image enhancement.
        hparams (dict): other kwargs for the AutoAugmentation scheme.
        transforms (dict): Strategies for transforming and enriching images.

    Returns:
        Class, enhanced operations performed.
    """
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [AugmentOp(
        name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in transforms]


class RandAugment:
    """
    Execute random enhancement strategy.

    Args:
        ops (str): Name of enhancement action.
        num_layers (int): The value of layers.
        choice_weights (bool): True means the same number can be taken, false means the same number cannot be taken.
    """
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        """
        No replacement when using weighted choice.

        Args:
            img (Image): Image to be processed.

        Returns:
            Image, processed image.
        """
        ops = np.random.choice(
            self.ops, self.num_layers, replace=self.choice_weights is None, p=self.choice_weights)
        for op in ops:
            img = op(img)
        return img


def rand_augment_transform(config_str, hparams):
    """
    Create a RandAugment transform.

    Args:
        config_str (str): String defining configuration of random augmentation.
        hparams (dict): Other kwargs for the RandAugmentation scheme.

    Returns:
        Class, a MindSpore compatible transform.
    """
    magnitude = _MAX_LEVEL
    num_layers = 2
    weight_idx = None
    transforms = _RAND_TRANSFORMS
    config = config_str.split('-')
    assert config[0] == 'rand'

    # [rand, m9, mstd0.5, inc1]
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param injected via hparams for now
            hparams.setdefault('magnitude_std', float(val))
        elif key == 'inc':
            if bool(val):
                transforms = _RAND_INCREASING_TRANSFORMS
        elif key == 'm':
            magnitude = int(val)
        elif key == 'n':
            num_layers = int(val)
        elif key == 'w':
            weight_idx = int(val)
        else:
            assert False, 'Unknown RandAugment config section'
    ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams, transforms=transforms)
    choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)


_AUGMIX_TRANSFORMS = [
    'AutoContrast',
    'ColorIncreasing',
    'ContrastIncreasing',
    'BrightnessIncreasing',
    'SharpnessIncreasing',
    'Equalize',
    'Rotate',
    'PosterizeIncreasing',
    'SolarizeIncreasing',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
]


def augmix_ops(magnitude=10, hparams=None, transforms=None):
    """
    Randomly enhance the image with different data, and then mix multiple data enhanced images.

    Args:
        magnitude (float): The probability of this operation and the amplitude of image enhancement.
        hparams (dict): other kwargs for the AutoAugmentation scheme.
        transforms (dict): Strategies for transforming and enriching images.

    Returns:
        Class, enhanced operations performed.
    """
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _AUGMIX_TRANSFORMS
    return [AugmentOp(
        name, prob=1.0, magnitude=magnitude, hparams=hparams) for name in transforms]


class AugMixAugment:
    """
    Randomly enhance the image with different data, and then mix multiple data enhanced images.

    Args:
        ops (str): Name of enhancement action.
        alpha (float): Learning rate parameter.
        width (int): The value of width.
        depth (int): The value of depth.
        blended (bool): Blended mode is faster but not well tested.
    """

    def __init__(self, ops, alpha=1., width=3, depth=-1, blended=False):
        self.ops = ops
        self.alpha = alpha
        self.width = width
        self.depth = depth
        self.blended = blended

    def _calc_blended_weights(self, ws, m):
        """
        Calculate blend weight values (blended mode is faster but not well tested).

        Args:
            ws (list): Value involved in calculation.
            m (int): Multiple value.

        Returns:
            float, the value of blended weights.
        """
        ws = ws * m
        cump = 1.
        rws = []
        for w in ws[::-1]:
            alpha = w / cump
            cump *= (1 - alpha)
            rws.append(alpha)
        return np.array(rws[::-1], dtype=np.float32)

    def _apply_blended(self, img, mixing_weights, m):
        """
        Apply a slightly faster mixing enhancement blended mode.

        Args:
            img (Image): Image to be processed.
            mixing_weights (list): Used to recalculate the mixing coefficient.
            m (int): Multiple value.

        Returns:
            Image, processed image.
        """
        img_orig = img.copy()
        ws = self._calc_blended_weights(mixing_weights, m)
        for w in ws:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            # no ops are in-place, deep copy not necessary
            img_aug = img_orig
            for op in ops:
                img_aug = op(img_aug)
            img = Image.blend(img, img_aug, w)
        return img

    def _apply_basic(self, img, mixing_weights, m):
        """
        Apply a blended mode without normalizations and PIL.

        Args:
            img (Image): Image to be processed.
            mixing_weights (list): Used to recalculate the mixing coefficient.
            m (int): Multiple value.

        Returns:
            Image, processed image.
        """
        img_shape = img.size[0], img.size[1], len(img.getbands())
        mixed = np.zeros(img_shape, dtype=np.float32)
        for mw in mixing_weights:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = img
            for op in ops:
                img_aug = op(img_aug)
            mixed += mw * np.asarray(img_aug, dtype=np.float32)
        np.clip(mixed, 0, 255., out=mixed)
        mixed = Image.fromarray(mixed.astype(np.uint8))
        return Image.blend(img, mixed, m)

    def __call__(self, img):
        """AugMixAugment apply"""
        mixing_weights = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        if self.blended:
            mixed = self._apply_blended(img, mixing_weights, m)
        else:
            mixed = self._apply_basic(img, mixing_weights, m)
        return mixed


def augment_and_mix_transform(config_str, hparams):
    """
    Create AugMix MindSpore transform.

    Args:
        config_str (str): String defining configuration of random augmentation.
        hparams (dict): Other kwargs for the RandAugmentation scheme.

    Returns:
        Class, a MindSpore compatible transform.
    """
    magnitude = 3
    width = 3
    depth = -1
    alpha = 1.
    blended = False
    hparams['magnitude_std'] = float('inf')
    config = config_str.split('-')
    assert config[0] == 'augmix'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param injected via hparams for now
            hparams.setdefault('magnitude_std', float(val))
        elif key == 'm':
            magnitude = int(val)
        elif key == 'w':
            width = int(val)
        elif key == 'd':
            depth = int(val)
        elif key == 'a':
            alpha = float(val)
        elif key == 'b':
            blended = bool(val)
        else:
            assert False, 'Unknown AugMix config section'
    ops = augmix_ops(magnitude=magnitude, hparams=hparams)
    return AugMixAugment(ops, alpha=alpha, width=width, depth=depth, blended=blended)
