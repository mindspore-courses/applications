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
"""test step for nerf"""

import time

import mindspore as ms
import numpy as np
from tqdm import tqdm

from engine import metrics
from nerf_utils.ray import generate_rays

__all__ = ["test_net"]


def test_net(img_h, img_w, focal, renderer, test_poses, gt=None, on_progress=None, on_complete=None):
    """
    Test the network and generate results.

    Args:
        img_h (int): Height of image plane.
        img_w (int): Width of image plane.
        focal (int): Focal length.
        renderer (Callable): The volume renderer.
        test_poses (Tensor): Poses used to test the network. shape (#poses, 4, 4).
        gt (Tensor, optional): The ground truth image. Default: None.
        on_progress (Callable, optional): A callback function invoked per generation of a result. Default: None.
        on_complete (Callable, optional): A callback function invoked after generating all results. Default: None.

    Returns:
        Tuple of 3 Tensor, the recorded values.

        - **time sequence** (ndarray) - The time value sequence.
        - **loss sequence** (ndarray) - The loss value sequence.
        - **psnr sequence** (ndarray) - The PSNR value sequence.
    """
    rgb_maps = []
    loss_ls = []
    psnr_ls = []
    time_ls = []

    reshape_op = ms.ops.Reshape()
    stack_op = ms.ops.Stack(axis=0)

    with tqdm(test_poses) as p_bar:
        for j, test_pose in enumerate(p_bar):
            t0 = time.time()

            # Generate rays for all pixels
            ray_origins, ray_dirs = generate_rays(img_h, img_w, focal, test_pose)
            ray_origins = reshape_op(ray_origins, (-1, 3))
            ray_dirs = reshape_op(ray_dirs, (-1, 3))
            test_batch_rays = stack_op([ray_origins, ray_dirs])

            # Retrieve testing results
            rgb_map, _ = renderer.inference(test_batch_rays)
            rgb_map = reshape_op(rgb_map, (img_h, img_w, 3))
            rgb_maps.append(rgb_map.asnumpy())

            # If given ground truth, compute MSE and PSNR
            if gt is not None:
                loss = metrics.mse(rgb_map, gt[j])
                psnr = metrics.psnr_from_mse(loss)
                loss_ls.append(float(loss))
                psnr_ls.append(float(psnr))

            time_ls.append(time.time() - t0)

            # Handle each testing result
            if on_progress:
                if isinstance(on_progress, list):
                    on_progress[0](j, rgb_maps[-1])
                    if gt is not None:
                        on_progress[1](j, gt[j].asnumpy())
                else:
                    on_progress(j, rgb_maps[-1])

    # Handle all testing results
    if on_complete:
        on_complete(np.stack(rgb_maps, 0))

    if not loss_ls:
        loss_ls = [0.0]
    if not psnr_ls:
        psnr_ls = [0.0]
    if not time_ls:
        time_ls = [0.0]

    return np.mean(time_ls), np.mean(loss_ls), np.mean(psnr_ls)
