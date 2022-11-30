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
"""blender data loader"""

import json
import os

import cv2
import imageio
import mindspore as md
import numpy as np

__all__ = ["load_blender_data"]


def trans_t(t):
    """
    Return the transformation in homogeneous coordinate form.

    Args:
        t (float): The transformation value.

    Returns:
        Tensor, the transformation matrix.
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]], dtype=np.float32)


def rot_phi(phi):
    """
    Return the transformation of the phi Euler angel in homogeneous coordinate form.

    Args:
        phi (float): The rotation value phi.

    Returns:
        Tensor, the rotation matrix.
    """
    return np.array([[1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0],
                     [0, np.sin(phi), np.cos(phi), 0], [0, 0, 0, 1]],
                    dtype=np.float32)


def rot_theta(th):
    """
    Return the transformation of theta Euler angel in homogeneous coordinate form.

    Args:
        th (float): The rotation value theta.

    Returns:
        Tensor, the rotation matrix.
    """
    return np.array([[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0],
                     [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]],
                    dtype=np.float32)


def pose_spherical(theta, phi, radius):
    """
    Return the spherical extrinsic camera poses for given theta, phi, and radius.

    Args:
        theta (float): The theta value.
        phi (float): The phi value.
        radius (float): The radius value.

    Returns:
        Tensor, the spherical rotation matrix.
    """
    c2w = trans_t(radius)
    c2w = np.matmul(rot_phi(phi / 180. * np.pi), c2w)
    c2w = np.matmul(rot_theta(theta / 180. * np.pi), c2w)
    c2w = np.matmul(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                 dtype=np.float32), c2w)
    return c2w


def load_blender_data(base_dir, half_res=False, test_skip=1):
    """
    Load blender data from structured directory.

    Args:
        base_dir (str): The base dir of the object data
        half_res (bool, optional): Whether to use half of the image resolution. Default: False.
        test_skip (int, optional): The number of skipped in-between test samples. Default: 1.

    Returns:
        Tuple of 5 terms, the input data.

        - **imgs** (Tensor) - The image tensors.
        - **poses** (Tensor) - The camera pose tensors.
        - **render_poses** (Tensor) - The render camera pose tensors.
        - **camera_intrinsics** (Tensor) - The camera intrinsics tensors.
        - **i_split** (Tensor) - The index of test samples.
    """
    splits = ["train", "val", "test"]
    meta_dict = {}
    for s in splits:
        with open(os.path.join(base_dir, f"transforms_{s}.json"), "r", encoding="utf-8") as fp:
            meta_dict[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = meta_dict[s]
        imgs = []
        poses = []
        if s == "train" or test_skip == 0:
            skip = 1
        else:
            skip = test_skip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(base_dir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))
        imgs = (np.array(imgs) / 255.0).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    cap_h, cap_w = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * cap_w / np.tan(0.5 * camera_angle_x)

    render_poses = np.stack([
        pose_spherical(angle, -30.0, 4.0)
        for angle in np.linspace(-180, 180, 40 + 1)[:-1]
    ], axis=0)

    if half_res:
        cap_h = cap_h // 2
        cap_w = cap_w // 2
        focal = focal / 2.0

        imgs_half_res = np.zeros((imgs.shape[0], cap_h, cap_w, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (cap_h, cap_w), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return_tuple = (
        md.Tensor(imgs).astype("float32"),
        md.Tensor(poses).astype("float32"),
        md.Tensor(render_poses).astype("float32"),
        [cap_h, cap_w, focal],
        i_split,
    )

    return return_tuple
