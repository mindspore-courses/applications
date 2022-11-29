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
"""
Helper that encapsulates the custom interface being used.
"""

from multiprocessing import Process

import numpy as np


def sync_trans(f):
    """
    Synchronous interface, multi process module of cross platform version.

    Args:
        f (optional): Object called by process instance.

    Returns:
        Process, a process class represents a process object.

    Raises:
        Exception: If the subprocess start failed.
    """
    def wrapper(*args, **kwargs):
        pro = Process(target=f, args=args, kwargs=kwargs)
        pro.start()
        return pro
    return wrapper


def check_obs_url(url: str):
    """
    Check whether the uniform resource locator of obs meets the requirements.

    Args:
        url (str): Url of obs.

    return:
        Bool, inspection results.
    """
    if url.startswith("s3") or url.startswith("obs"):
        return True
    return False


def get_2d_sin_cos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Get 2d sin-cos pos embedding.

    Args:
        embed_dim (int): Dimension of embedding.
        grid_size (int): Int of the grid height and width.
        cls_token (bool): Whether to use cls_token.

    return:
        [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token).
    """

    # Get the length and width of each small patch
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sin_cos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sin_cos_pos_embed_from_grid(embed_dim, grid):
    """
    Get 2d sin-cos pos embedding from grid.

    Args:
        embed_dim (int): Dimension of embedding.
        grid (list): Int of the grid height and width.

    return:
        [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token).
    """
    assert embed_dim % 2 == 0

    # Get two one-dimensional position codes
    emb_h = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    # Splice the two one-dimensional position codes together to get the overall position code.
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sin_cos_pos_embed_from_grid(embed_dim, pos):
    """
    Get 1d sin-cos pos_embed from grid.

    Args:
        embed_dim (int): Output dimension for each position.
        pos (list): A list of positions to be encoded: size (M,).

    Returns:
        List, the shape is (M, D).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
