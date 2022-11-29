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
""" Custom transform operation for sceneflow dataset. """
import random
import numpy as np
import cv2
from PIL import Image

from mindspore import Tensor
import mindspore.dataset.vision.py_transforms as py_vision

class ToTensor:
    """
    Resize sceneflow data

    Inputs:
        sample (dict): sceneflow data sample.

    Outputs:
        dict: tensor sceneflow data sample.

    Examples:
        >>> to_tensor = ToTensor()
        >>> x = to_tensor(x)
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        return {
            "frame": py_vision.ToTensor()(sample["frame"]),
            "pre_frame": py_vision.ToTensor()(sample["pre_frame"]),
            "optical_flow": Tensor.from_numpy(sample["optical_flow"]),
            "reverse_optical_flow": Tensor.from_numpy(sample["reverse_optical_flow"]),
            "motion_boundaries": Tensor.from_numpy(np.array(sample["motion_boundaries"]).astype(np.bool))
        }

class Resize:
    """
    Resize sceneflow data

    Args:
        new_width (int): new width for image.
        new_height (int): new height for image.

    Inputs:
        sample (dict): sceneflow sample.

    Outputs:
        dict: resized sample.

    Examples:
        >>> resize = Resize(640, 320)
        >>> x = resize(x)
    """
    def __init__(self, new_width, new_height):
        self.new_width = new_width
        self.new_height = new_height

    def resize_image(self, image):
        """Resize image"""
        return image.resize((self.new_width, self.new_height))

    def resize_optical_flow(self, optical_flow):
        """Resize optical flow"""
        orig_height, orig_width = optical_flow.shape[:2]
        optical_flow_resized = cv2.resize(optical_flow, (self.new_width, self.new_height))
        h_scale, w_scale = self.new_height / orig_height, self.new_width / orig_width
        optical_flow_resized[..., 0] *= w_scale
        optical_flow_resized[..., 1] *= h_scale
        return optical_flow_resized

    def __call__(self, sample):
        return {
            "frame": self.resize_image(sample["frame"]),
            "pre_frame": self.resize_image(sample["pre_frame"]),
            "optical_flow": self.resize_optical_flow(sample["optical_flow"]),
            "reverse_optical_flow": self.resize_optical_flow(sample["reverse_optical_flow"]),
            "motion_boundaries": self.resize_image(sample["motion_boundaries"])
        }


class RandomHorizontalFlip:
    """
    Horizontal flip sceneflow data by probability

    Args:
        p (float32): probability of random horizontal operation.

    Inputs:
        sample (dict): sceneflow data sample.

    Outputs:
        dict: Horizontal-fliped sceneflow data sample.

    Examples:
        >>> rhf = RandomHorizontalFlip(p=0.5)
        >>> x = rhf(x)
    """
    def __init__(self, p=0.5):
        self.p = p

    @staticmethod
    def flip_image(image):
        """Flip image"""
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    @staticmethod
    def flip_optical_flow(optical_flow):
        """Flip optical flow"""
        optical_flow = np.flip(optical_flow, axis=1).copy()
        optical_flow[..., 0] *= -1
        return optical_flow

    def __call__(self, sample):
        if random.random() < self.p:
            return {
                "frame": self.flip_image(sample["frame"]),
                "pre_frame": self.flip_image(sample["pre_frame"]),
                "optical_flow": self.flip_optical_flow(sample["optical_flow"]),
                "reverse_optical_flow": self.flip_optical_flow(sample["reverse_optical_flow"]),
                "motion_boundaries": self.flip_image(sample["motion_boundaries"])
            }
        return sample
