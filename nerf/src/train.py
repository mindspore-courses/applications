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
"""Buile and train model."""

import os
import time

import mindspore as md
import numpy as np
from engine import RendererWithCriterion, test_net, train_net
from tqdm import tqdm

from data import load_blender_data, load_llff_data
from models import VolumeRenderer
from nerf_utils.config import get_config
from nerf_utils.engine_utils import context_setup, create_nerf
from nerf_utils.ray import generate_rays
from nerf_utils.results_handler import save_image, save_video
from nerf_utils.sampler import sample_grid_2d


def train_pipeline(config, out_dir):
    """
    Train nerf model: data preparation, model and optimizer preparation, and model training.

    Args:
        config (Config): The config object.
        out_dir (str): The output directory.
    """
    md.set_seed(1)

    print(">>> Loading dataset")

    if config.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(config.data_dir, config.half_res,
                                                                      config.test_skip)
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
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
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

    # Create network models, optimizer and renderer
    print(">>> Creating models")

    # Create nerf model
    (
        start_iter,
        optimizer,
        model_coarse,
        model_fine,
        embed_fn,
        embed_dirs_fn,
    ) = create_nerf(config, out_dir)
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

    renderer_with_criterion = RendererWithCriterion(renderer)
    optimizer = md.nn.Adam(
        params=renderer.trainable_params(),
        learning_rate=config.l_rate,
        beta1=0.9,
        beta2=0.999,
    )

    if config.precision == "fp16":
        loss_scale = 1024.0
        train_renderer = CustomTrainOneStepCell(renderer_with_criterion, optimizer, loss_scale)
    else:
        train_renderer = md.nn.TrainOneStepCell(renderer_with_criterion, optimizer)
    train_renderer.set_train()

    # Start training
    print(">>> Start training")

    cap_n_rand = config.cap_n_rand

    # Move training data to GPU
    images = md.Tensor(images)
    poses = md.Tensor(poses)

    # Maximum training iterations
    cap_n_iters = config.cap_n_iters
    if start_iter >= cap_n_iters:
        return

    train_model(config, out_dir, images, poses, i_train, i_test, cap_h, cap_w, focal, start_iter, optimizer,
                global_steps, renderer, train_renderer, cap_n_rand, cap_n_iters)


def train_model(config, out_dir, images, poses, i_train, i_test, cap_h, cap_w, focal, start_iter, optimizer,
                global_steps, renderer, train_renderer, cap_n_rand, cap_n_iters):
    """
    Training model iteratively.

    Args:
        config (Config): The config object.
        out_dir (str): The output directory.
        images (Tensor): The image tensors.
        poses (Tensor): The extrinsic camera poses.
        i_train (Tensor): The index tensor for train.
        i_test (Tensor): The index tensor for test.
        cap_h (int): The image height.
        cap_w (int): The image width.
        focal (int): The focal length.
        start_iter (int): The start training iteration step.
        optimizer (nn.Cell): The optimizer from MindSpore.
        global_steps (int): The global training step.
        renderer (nn.Cell): The volume renderer for radiance fields.
        train_renderer (nn.Cell): The trainer for renderer.
        cap_n_rand (int): The number of sampled rays.
        cap_n_iters (int): The number of training iterations.
    """
    with tqdm(range(1, cap_n_iters + 1)) as p_bar:
        p_bar.n = start_iter

        for _ in p_bar:
            # Show progress
            p_bar.set_description(f"Iter {global_steps + 1:d}")
            p_bar.update()

            # Start time of the current iteration
            time_0 = time.time()

            img_i = int(np.random.choice(i_train))

            target = images[img_i]
            pose = poses[img_i, :3, :4]

            if cap_n_rand is not None:
                rays_o, rays_d = generate_rays(cap_h, cap_w, focal,
                                               md.Tensor(pose))  # (cap_h, cap_w, 3), (cap_h, cap_w, 3)
                sampled_rows, sampled_cols = sample_grid_2d(cap_h, cap_w, cap_n_rand)
                rays_o = rays_o[sampled_rows, sampled_cols]  # (cap_n_rand, 3)
                rays_d = rays_d[sampled_rows, sampled_cols]  # (cap_n_rand, 3)

                batch_rays = md.ops.Stack(axis=0)([rays_o, rays_d])
                target_s = target[sampled_rows, sampled_cols]  # (cap_n_rand, 3)

            loss, psnr = train_net(config, global_steps, train_renderer, optimizer, batch_rays, target_s)

            p_bar.set_postfix(time=time.time() - time_0, loss=loss, psnr=psnr)

            # Logging
            # Save training states
            if (global_steps + 1) % config.i_ckpt == 0:
                path = os.path.join(out_dir, f"{global_steps + 1:06d}.tar")

                md.save_checkpoint(
                    save_obj=renderer,
                    ckpt_file_name=path,
                    append_dict={"global_steps": global_steps},
                    async_save=True,
                )
                p_bar.write(f"Saved checkpoints at {path}")

            # Save testing results
            if (global_steps + 1) % config.i_testset == 0:
                test_save_dir = os.path.join(out_dir, f"test_{global_steps + 1:06d}")
                os.makedirs(test_save_dir, exist_ok=True)

                p_bar.write(f"Testing (iter={global_steps + 1}):")

                test_time, test_loss, test_psnr = test_net(
                    cap_h,
                    cap_w,
                    focal,
                    renderer,
                    md.Tensor(poses[i_test.tolist()]),
                    images[i_test.tolist()],
                    on_progress=lambda j, img: save_image(j, img, test_save_dir),  # pylint: disable=cell-var-from-loop
                    on_complete=lambda imgs: save_video(global_steps + 1, imgs, test_save_dir),  # pylint: disable=cell-var-from-loop
                )

                p_bar.write(
                    f"Testing results: [ Mean Time: {test_time:.4f}s, Loss: {test_loss:.4f}, PSNR: {test_psnr:.4f} ]")

            global_steps += 1


grad_scale = md.ops.MultitypeFuncGraph("grad_scale")
@grad_scale.register("Tensor", "Tensor")
def gradient_scale(scale, grad):
    """
    Helper function for gradient scaling.

    Args:
        scale (Tensor): scaler.
        grad (Tensor): gradients.

    Returns:
        Tensor, scaled gradients.
    """
    return grad * md.ops.cast(scale, md.ops.dtype(grad))

class CustomTrainOneStepCell(md.nn.TrainOneStepCell):
    """
    Volume Renderer architecture.

    Args:
        network (nn.Cell): The network to be half precision.
        optimizer (nn.Optimizer): The optimizer to be half precision.
        sens (float, optional): The reciprocal sense float. Defaults to 1.0.

    Inputs:
        - **inputs** (Tensor) - The model inputs.

    Outputs:
        Tensor, the output tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(CustomTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.hyper_map = md.ops.HyperMap()
        self.reciprocal_sense = md.Tensor(1 / sens, md.float32)

    def scale_grad(self, gradients):
        """Scaling gradient."""
        gradients = self.hyper_map(md.ops.partial(grad_scale, self.reciprocal_sense), gradients)
        return gradients

    def construct(self, *inputs):
        """Construct model."""
        loss = self.network(*inputs)
        sens = md.ops.fill(loss.dtype, loss.shape, self.sens)
        # calculate gradients, the sens will equal to the loss_scale
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        # gradients / loss_scale
        grads = self.scale_grad(grads)
        # reduce gradients in distributed scenarios
        grads = self.grad_reducer(grads)
        loss = md.ops.depend(loss, self.optimizer(grads))
        return loss


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

    # Start training pipeline
    train_pipeline(config, out_dir)


if __name__ == "__main__":
    main()
