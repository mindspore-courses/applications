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
"""llff data loader"""

import os
from subprocess import check_output

import imageio
import mindspore as md
import numpy as np

__all__ = ["load_llff_data"]


def _minify(base_dir, factors=(), resolutions=()):
    """
    Save and process images for lower resolution.

    Args:
        base_dir (str): The base directory.
        factors (tuple, optional): The re-factor scales. Default: ().
        resolutions (tuple, optional): The target resolutions. Default: ().
    """
    need_to_load = False
    for r in factors:
        img_dir = os.path.join(base_dir, f"images_{r}")
        if not os.path.exists(img_dir):
            need_to_load = True
    for r in resolutions:
        img_dir = os.path.join(base_dir, f"images_{r[1]}x{r[0]}")
        if not os.path.exists(img_dir):
            need_to_load = True
    if not need_to_load:
        return

    img_dir = os.path.join(base_dir, "images")
    imgs = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
    imgs = [f for f in imgs if any((f.endswith(ex) for ex in ("JPG", "jpg", "png", "jpeg", "PNG")))]
    img_dir_orig = img_dir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = f"images_{r}"
            resize_arg = f"{100.0 / r}%"
        else:
            name = f"images_{r[1]}x{r[0]}"
            resize_arg = f"{r[1]}x{r[0]}"
        img_dir = os.path.join(base_dir, name)
        if os.path.exists(img_dir):
            continue

        print("Minifying", r, base_dir)

        os.makedirs(img_dir)
        check_output(f"cp {img_dir_orig}/* {img_dir}", shell=True)

        ext = imgs[0].split(".")[-1]
        args = " ".join(["mogrify", "-resize", resize_arg, "-format", "png", f"*.{ext}"])
        print(args)
        os.chdir(img_dir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != "png":
            check_output(f"rm {img_dir}/*.{ext}", shell=True)
            print("Removed duplicates")
        print("Done")


def _load_data(base_dir, factor=None, width=None, height=None, load_imgs=True):
    """
    Load images after applying resolution minification.

    Args:
        base_dir (str): The base directory.
        factor (tuple, optional): The re-factor scales. Default: None.
        width (int, optional): The image width. Default: None.
        height (int, optional): The image height. Default: None.
        load_imgs (bool, optional): Whether to load images or not. Default: True.

    Returns:
        Tuple of 3 Tensor, the input tensors.

        - **poses** (Tensor) - The LLFF extrinsic poses.
        - **bds** (Tensor) - The near and far bounds.
        - **imgs** (Tensor) - The input image tensors.
    """
    poses_arr = np.load(os.path.join(base_dir, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [
        os.path.join(base_dir, "images", f)
        for f in sorted(os.listdir(os.path.join(base_dir, "images")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][0]
    sh = imageio.imread(img0).shape

    sfx = ""

    if factor is not None:
        sfx = f"_{factor}"
        _minify(base_dir, factors=[factor])
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(base_dir, resolutions=[[height, width]])
        sfx = f"_{width}x{height}"
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(base_dir, resolutions=[[height, width]])
        sfx = f"_{width}x{height}"
    else:
        factor = 1

    img_dir = os.path.join(base_dir, "images" + sfx)
    if not os.path.exists(img_dir):
        print(img_dir, "does not exist, returning")
        return None, None, None

    img_files = [
        os.path.join(img_dir, f)
        for f in sorted(os.listdir(img_dir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]
    if poses.shape[-1] != len(img_files):
        print(f"Mismatch between imgs {len(img_files)} and poses {poses.shape[-1]} !!!!")
        return None, None, None

    sh = imageio.imread(img_files[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor

    if not load_imgs:
        return poses, bds, None

    def imread(f):
        if f.endswith("png"):
            return imageio.imread(f, ignoregamma=True)
        return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255.0 for f in img_files]
    imgs = np.stack(imgs, -1)

    print("Loaded image data", imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def normalize(x):
    """
    Normalize x to [-1, 1].

    Args:
        x (Tensor): The un-normalized tensor.

    Returns:
        Tensor, the normalized tensor.
    """
    return x / np.linalg.norm(x)


def view_matrix(z, up, pos):
    """
    Get view matrix obeying OpenGL conversion.

    Args:
        z (Tensor): The z-axis direction vector.
        up (Tensor): The up direction vector.
        pos (Tensor): The position matrix.

    Returns:
        Tensor, the view camera matrix.
    """
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def pts_to_cam(pts, c2w):
    """
    Transform points from world to camera coordinates.

    Args:
        pts (Tensor): The points in world coordinates.
        c2w (Tensor): camera to world transformation.

    Returns:
        Tensor, points in camera coordinates.
    """
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    """
    Compute average camera pose.

    Args:
        poses (Tensor): The camera poses.

    Returns:
        Tensor, the average poses.
    """
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([view_matrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, z_rate, rots, cap_n):
    """
    Recenter poses to origin from COLMAP cameras.

    Args:
        c2w (Tensor): The camera to world matrix.
        up (Tensor): The up vector.
        rads (Tensor): The rad vector.
        focal (Tensor): The focal length.
        z_rate (Tensor): The z rate value.
        rots (Tensor): The rots value.
        cap_n (Tensor): The number of cameras.

    Returns:
        Tensor, the spiral camera matrix.
    """
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, cap_n + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * z_rate), 1.0]) * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([view_matrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):
    """
    Recenter poses to origin from COLMAP cameras.

    Args:
        poses (Tensor): The un-centered poses.

    Returns:
        Tensor, the re-centered poses.
    """
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses, bds):
    """
    Generate spherify poses from COLMAP cameras.

    Args:
        poses (Tensor): The original poses.
        bds (Tensor): The near and far bounds.

    Returns:
        Tuple of 3 tensor, the spherical camera poses.

        - **poses_reset** (Tensor) - The reposed camera matrix.
        - **new_poses** (Tensor) - The re-centered camera matrix.
        - **bds** (Tensor) - The re-centered near and far bounds.
    """
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        cap_a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -cap_a_i @ rays_o
        pt_min_dist = np.squeeze(-np.linalg.inv((np.transpose(cap_a_i, [0, 2, 1]) @ cap_a_i).mean(0)) @ (b_i).mean(0))
        return pt_min_dist

    pt_min_dist = min_line_dist(rays_o, rays_d)

    center = pt_min_dist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    rad_circle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):

        cam_origin = np.array([rad_circle * np.cos(th), rad_circle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(cam_origin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = cam_origin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds


def load_llff_data(base_dir, factor=8, recenter=True, bd_factor=0.75, spherify=False, path_z_flat=False):
    """
    Load LLFF data.

    Args:
        base_dir (str): Base directory of LLFF data.
        factor (int, optional): Factor of LLFF data. Default: 8.
        recenter (bool, optional): Recenter poses to origin. Default: True.
        bd_factor (float, optional): Factor of bounding box. Default: 0.75.
        spherify (bool, optional): Spherify poses. Default: False.
        path_z_flat (bool, optional): Path z is flat. Default: False.

    Returns:
        Tuple of 5 tensors, the input data.

        - **images** (Tensor) - The processed input image tensors.
        - **poses** (Tensor) - Pose of camera.
        - **bds** (Tensor) - The near and far bounds.
        - **render_poses** (Tensor) - The rendering spherified camera poses.
        - **i_test** (Tensor) - The index of test samples.
    """
    poses, bds, imgs = _load_data(base_dir, factor=factor)  # factor=8 down-samples original imgs by 8x
    print("Loaded", base_dir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)
    else:
        c2w = poses_avg(poses)
        print("re-centered", c2w.shape)
        print(c2w[:3, :4])

        # Get spiral: average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        cap_n_views = 120
        cap_n_rots = 2
        if path_z_flat:
            z_loc = -close_depth * 0.1
            c2w_path[:3, 3] = c2w_path[:3, 3] + z_loc * c2w_path[:3, 2]
            rads[2] = 0.0
            cap_n_rots = 1
            cap_n_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, z_rate=0.5, rots=cap_n_rots, cap_n=cap_n_views)

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print("Data:")
    print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print("HOLDOUT view is", i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return_tuple = (
        md.Tensor(images).astype("float32"),
        md.Tensor(poses).astype("float32"),
        bds,
        md.Tensor(render_poses).astype("float32"),
        i_test,
    )

    return return_tuple
