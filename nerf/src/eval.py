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
"""Buile and eval model."""

import os

import mindspore as md
import numpy as np
from engine import test_net

from data import load_blender_data, load_llff_data
from models import VolumeRenderer
from nerf_utils.config import get_config
from nerf_utils.engine_utils import context_setup, create_nerf
from nerf_utils.results_handler import save_image, save_video


def eval_pipeline(config, out_dir):
    """
    Eval nerf model.

    Args:
        config (Config): The config object.
        out_dir (str): The output directory.
    """
    md.set_seed(1)

    print(">>> Loading dataset")

    if config.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(config.data_dir, config.half_res,
                                                                      config.test_skip)
        render_poses = render_poses[:, :3, :4]
        print("Loaded blender", images.shape, render_poses.shape, hwf, config.data_dir)
        i_train, i_val, i_test = i_split
        near = 2.0
        far = 6.0

        if config.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif config.dataset_type == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            config.data_dir,
            config.factor,
            recenter=True,
            bd_factor=0.75,
            spherify=config.spherify,
        )
        if config.render_test:
            hwf = poses[0, :3, -1]
        else:
            hwf = render_poses[0, :3, -1]
        poses = poses[:, :3, :4]
        render_poses = render_poses[:, :3, :4]

        print("Loaded llff", images.shape, render_poses.shape, hwf, config.data_dir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if config.llff_hold > 0:
            print("Auto LLFF holdout,", config.llff_hold)
            i_test = np.arange(images.shape[0])[::config.llff_hold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])

        print("DEFINING BOUNDS")
        config.no_ndc = True
        if config.no_ndc:
            near = float(np.min(bds)) * 0.9
            far = float(np.max(bds)) * 1.0
        else:
            near = 0.0
            far = 1.0
        print("NEAR FAR", near, far)

    else:
        print("Unknown dataset type", config.dataset_type, "exiting")
        return

    if config.render_test:
        render_poses = poses[i_test.tolist()]

    print(f"TRAIN views: {i_train}\nTEST views: {i_test}\nVAL views: {i_val}")

    # Cast intrinsics to right types
    cap_h, cap_w, focal = hwf
    cap_h, cap_w = int(cap_h), int(cap_w)

    hwf = [cap_h, cap_w, focal]
    # Setup logging and directory for results
    print(">>> Saving checkpoints and results in", out_dir)
    # Create output directory if not existing

    os.makedirs(out_dir, exist_ok=True)
    # Record current configuration
    with open(os.path.join(out_dir, "configs.txt"), "w+", encoding="utf-8") as config_f:
        attrs = vars(config)
        for k in attrs:
            config_f.write(f"{k} = {attrs[k]}\n")

    # Create network models and renderer
    print(">>> Creating models")

    # Create nerf model
    start_iter, _, model_coarse, model_fine, embed_fn, embed_dirs_fn = create_nerf(config, out_dir)
    # Training steps
    global_steps = start_iter
    # Create volume renderer
    renderer = VolumeRenderer(
        config.chunk,
        config.cap_n_samples,
        config.cap_n_importance,
        config.net_chunk,
        config.white_bkgd,
        model_coarse,
        model_fine,
        embed_fn,
        embed_dirs_fn,
        near,
        far,
    )

    if config.precision == "fp16":
        print("use fp16.")
        renderer.model_coarse.to_float(md.dtype.float16)
        if renderer.model_fine is not None:
            renderer.model_fine.to_float(md.dtype.float16)

        images = images.astype(md.dtype.float16)
        poses = poses.astype(md.dtype.float16)
        render_poses = render_poses.astype(md.dtype.float16)

    # Only render results by pre-trained models
    print(">>> Render only")

    # Move testing data to GPU
    render_poses = md.Tensor(render_poses)

    # Path to save rendering results
    render_save_dir = os.path.join(out_dir, f"renderonly_{global_steps:06d}")
    gt_save_dir = os.path.join(out_dir, f"renderonly_gt_{global_steps:06d}")
    os.makedirs(render_save_dir, exist_ok=True)
    os.makedirs(gt_save_dir, exist_ok=True)

    save_img_fn = [
        lambda j, img: save_image(j, img, render_save_dir),
        lambda j, img: save_image(j, img, gt_save_dir),
    ]
    if config.generate_video:
        save_video_fn = lambda imgs: save_video(global_steps, imgs, render_save_dir)
    else:
        save_video_fn = None

    print(f"Rendering (iter={global_steps}):")

    test_time, test_loss, test_psnr = test_net(
        cap_h,
        cap_w,
        focal,
        renderer,
        render_poses,
        images[i_test.tolist()] if config.render_test else None,
        on_progress=save_img_fn,
        on_complete=save_video_fn,
    )

    if config.render_test:
        print(f"Testing results: [ Mean Time: {test_time:.4f}s, Loss: {test_loss:.4f}, PSNR: {test_psnr:.4f} ]")
    return


def main():
    """Main function, set up config."""
    config = get_config()

    # Set up running device
    context_setup(config.device_id, config.device, getattr(md.context, config.mode))

    # Output directory
    base_dir = config.base_dir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Experiment name
    exp_name = config.dataset_type + "_" + config.name
    # Get the experiment number
    exp_num = max([int(fn.split("_")[-1]) for fn in os.listdir(base_dir) if fn.find(exp_name) >= 0] + [0])
    if config.no_reload:
        exp_num += 1

    # Output directory
    out_dir = os.path.join(base_dir, exp_name + "_" + str(exp_num))

    # Start eval pipeline
    eval_pipeline(config, out_dir)


if __name__ == "__main__":
    main()
