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
"""build config"""

import argparse

__all__ = ["get_config"]


def get_config():
    """Set up config."""
    parser = argparse.ArgumentParser()
    # General options
    parser.add_argument("--name", required=True, type=str, help="experiment name")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./results/",
        help="where to store ckpts and results",
    )
    parser.add_argument("--data_dir", type=str, default="./data/llff/fern", help="input data directory")
    # Training options
    parser.add_argument("--cap_n_iters", type=int, default=200000, help="max training iterations")
    parser.add_argument("--net_depth", type=int, default=8, help="layers in network")
    parser.add_argument("--net_width", type=int, default=256, help="channels per layer")
    parser.add_argument("--net_depth_fine", type=int, default=8, help="layers in fine network")
    parser.add_argument(
        "--net_width_fine",
        type=int,
        default=256,
        help="channels per layer in fine network",
    )
    parser.add_argument(
        "--cap_n_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--l_rate", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--l_rate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000 steps)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 32,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--net_chunk",
        type=int,
        default=1024 * 64,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--no_batching",
        action="store_true",
        help="only take random rays from 1 image at a time",
    )
    parser.add_argument("--no_reload", action="store_true", help="do not reload weights from saved ckpt")
    parser.add_argument("--device_id", type=int, default=0, help="the index of device")
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=("GPU", "CPU", "Ascend"),
        help="the index of gpu device",
    )
    # Rendering options
    parser.add_argument("--cap_n_samples", type=int, default=64, help="number of coarse samples per ray")
    parser.add_argument(
        "--cap_n_importance",
        type=int,
        default=0,
        help="number of additional fine samples per ray",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument("--use_view_dirs", action="store_true", help="use full 5D input instead of 3D")
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--multi_res",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multi_res_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )
    parser.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="down- sampling factor to speed up rendering, set 4 or 8 for fast preview",
    )
    # Dataset options
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="llff",
        help="options: llff / blender / deepvoxels",
    )
    parser.add_argument(
        "--test_skip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )
    # DeepVoxels flags
    parser.add_argument(
        "--shape",
        type=str,
        default="greek",
        help="options : armchair / cube / greek / vase",
    )
    # Blender flags
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for deepvoxels)",
    )
    parser.add_argument(
        "--half_res",
        action="store_true",
        help="load blender synthetic data at 400x400 instead of 800x800",
    )
    # LLFF flags
    parser.add_argument("--factor", type=int, default=8, help="down sample factor for LLFF images")
    parser.add_argument(
        "--no_ndc",
        action="store_true",
        help="do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    parser.add_argument(
        "--lin_disp",
        action="store_true",
        help="sampling linearly in disparity rather than depth",
    )
    parser.add_argument("--spherify", action="store_true", help="set for spherical 360 scenes")
    parser.add_argument(
        "--llff_hold",
        type=int,
        default=8,
        help="will take every 1/N images as LLFF test set, paper uses 8",
    )
    # Logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=100,
        help="frequency of console printout and metric logging",
    )
    parser.add_argument("--i_img", type=int, default=500, help="frequency of tensorboard image logging")
    parser.add_argument("--i_ckpt", type=int, default=10000, help="frequency of ckpt saving")
    parser.add_argument("--i_testset", type=int, default=20000, help="frequency of testset saving")
    parser.add_argument("--ckpt", type=str, default=None, help="the path of external checkpoint")
    parser.add_argument(
        "--mode",
        type=str,
        default="GRAPH_MODE",
        choices=("PYNATIVE_MODE", "GRAPH_MODE"),
        help="the path of external checkpoint",
    )
    parser.add_argument("--precision",
                        type=str,
                        default="fp16",
                        choices=("fp32", "fp16"),
                        help="the model precision, support fp32 or fp16")
    parser.add_argument("--generate_video", action="store_true", help="generate rendering video.")
    return parser.parse_args()
