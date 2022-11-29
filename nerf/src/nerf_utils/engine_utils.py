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
"""nerf engine utils"""

import os
import warnings

import mindspore as md

from models import NeRFMLP
from nerf_utils import Embedder

__all__ = ["get_embedder", "create_nerf", "context_setup"]


def get_embedder(multi_res, i=0):
    """
    Get embedder function.

    Args:
        multi_res (int): Log2 of max freq for positional encoding.
        i (int, optional): Set 0 for default positional encoding, -1 for none. Default: 0.

    Returns:
        Tuple of nn.Cell and int, embedder and the output dimensions.

        - **embedder** (nn.Cell) - The embedder.
        - **out_dims** (int) - The output dimensions.
    """
    if i == -1:
        return md.ops.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_pow": multi_res - 1,
        "num_freqs": multi_res,
        "log_sampling": True,
        "periodic_fns": [md.ops.Sin(), md.ops.Cos()],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = embed_kwargs
    return embed, embedder_obj.out_dims


def create_nerf(config, out_dir):
    """
    Create nerf model and load weights.

    Args:
        config (Config): The config object.
        out_dir (str): The output directory.

    Returns:
        Tuple of 6 items. The network items.

        - **start_iter** (int) - The start iteration number.
        - **optimizer** (Cell) - The MLP optimizer.
        - **model_coarse** (Cell) - The coarse MLP.
        - **model_fine** (Cell) - The fine MLP.
        - **embed_fn** (Cell) - The positional embedder functions for location.
        - **embed_dirs_fn** (Cell) - The positional embedder functions for direction.
    """
    embed_fn, input_ch = get_embedder(config.multi_res, config.i_embed)

    input_ch_views = 0
    embed_dirs_fn = None
    if config.use_view_dirs:
        embed_dirs_fn, input_ch_views = get_embedder(config.multi_res_views, config.i_embed)
    # Create networks
    output_ch = 4
    skips = [4]
    model_coarse = NeRFMLP(
        cap_d=config.net_depth,
        cap_w=config.net_width,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_view_dirs=config.use_view_dirs,
    )
    grad_vars = [{"params": model_coarse.trainable_params()}]

    model_fine = None
    if config.cap_n_importance > 0:
        model_fine = NeRFMLP(
            cap_d=config.net_depth_fine,
            cap_w=config.net_width_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_view_dirs=config.use_view_dirs,
        )
        grad_vars += [{"params": model_fine.trainable_params()}]

    optimizer = None
    # Load checkpoints
    start_iter = 0
    if config.ckpt is not None:
        ckpts = [config.ckpt]
    else:
        ckpts = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if ".tar" in f]

    print("Found ckpts", ckpts)
    if (ckpts and not config.no_reload) or config.ckpt is not None:
        # Reload the latest ckpt
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = md.load_checkpoint(ckpt_path)

        # Load training steps
        start_iter = int(ckpt["global_steps"]) + 1

        # Load network weights
        md.load_param_into_net(
            model_coarse,
            {key: value for key, value in ckpt.items() if ".model_coarse." in key},
        )
        if model_fine is not None:
            md.load_param_into_net(
                model_fine,
                {key: value for key, value in ckpt.items() if ".model_fine." in key},
            )
    else:
        print("No ckpt reloaded")

    return start_iter, optimizer, model_coarse, model_fine, embed_fn, embed_dirs_fn


def context_setup(idx, device, mode):
    """
    Set up running context.

    Args:
        idx (int): The device index.
        device (str): The target platforms. ``Ascend`` or ``GPU``.
        mode (str): The running mode. `PYNATIVE_MODE` or `GRAPH_MODE`.

    Raises:
        NotImplementedError: `cumprod` ops does not support CPU.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if device == "CPU":
        raise NotImplementedError("`cumprod` ops does not support CPU.")

    md.context.set_context(mode=mode, device_target=device, device_id=idx)

    warnings.warn("Not support N-D searchsorted, set `max_call_depth=20000` to prevent `for loop` collapse.")
    md.context.set_context(max_call_depth=20000)
