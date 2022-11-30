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
"""nerf ray generator"""

import mindspore as md

__all__ = ["generate_rays"]


def generate_rays(h, w, f, pose):
    """
    Given an image plane, generate rays from the camera origin to each pixel on the image plane.

    Args:
        h (int): Height of the image plane.
        w (int): Width of the image plane.
        f (int): Focal length of the image plane.
        pose (Tensor): The extrinsic parameters of the camera. (3, 4) or (4, 4).

    Returns:
        Tuple of 2 tensor, origins of rays and directions of rays.

        - **rays_origins** (Tensor) - Origins of rays.
        - **ray_dirs** (Tensor) - Directions of rays.
    """

    # Coordinates of the 2D grid
    f = md.Tensor(f, dtype=md.float32)
    cols = md.ops.ExpandDims()(md.numpy.linspace(-1.0 * w / 2, w - 1 - w / 2, w) / f, 0).repeat(h, axis=0)
    rows = md.ops.ExpandDims()(-1.0 * md.numpy.linspace(-1.0 * h / 2, h - 1 - h / 2, h) / f, 1).repeat(w, axis=1)

    # Ray directions for all pixels
    ray_dirs = md.numpy.stack([cols, rows, -1.0 * md.numpy.ones_like(cols)], axis=-1)
    # Apply rotation transformation to make each ray orient according to the camera
    unsqueeze_op = md.ops.ExpandDims()
    ray_dirs = md.numpy.sum(unsqueeze_op(ray_dirs, 2) * pose[:3, :3], axis=-1)
    # Origin position
    rays_origins = pose[:3, -1].expand_as(ray_dirs)

    return rays_origins, ray_dirs.astype(pose.dtype)
