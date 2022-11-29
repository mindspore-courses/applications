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
"""ray sampler for nerf"""

import mindspore as md
import numpy as np

__all__ = ["sample_grid_2d", "sample_along_rays", "sample_pdf"]


def sample_grid_2d(cap_h, cap_w, cap_n):
    """
    Sample cells in an cap_h x cap_w mesh grid.

    Args:
        cap_h (int): Height of the mesh grid.
        cap_w (int): Width of the mesh grid.
        cap_n (int): The number of samples.

    Returns:
        Tuple of 2 Tensor, sampled rows and sampled columns.

        - **select_coords_x** (Tensor) - Sampled rows.
        - **select_coords_y** (Tensor) - Sampled columns.
    """
    if cap_n > cap_w * cap_h:
        cap_n = cap_w * cap_h

    # Create a 2D mesh grid where each element is the coordinate of the cell
    stack_op = md.ops.Stack(-1)
    coords = stack_op(
        md.numpy.meshgrid(
            md.numpy.linspace(0, cap_h - 1, cap_h),
            md.numpy.linspace(0, cap_w - 1, cap_w),
            indexing="ij",
        ))
    # Flat the mesh grid
    coords = md.ops.Reshape()(coords, (-1, 2))
    # Sample N cells in the mesh grid
    select_indexes = np.random.choice(coords.shape[0], size=[cap_n], replace=False)
    # Sample N cells among the mesh grid
    select_coords = coords[select_indexes.tolist()].astype("int32")

    return select_coords[:, 0], select_coords[:, 1]


def sample_along_rays(near, far, cap_cap_n_samples, lin_disp=False, perturb=True):
    """
    Sample points along rays.

    Args:
        near (Tensor): A vector containing nearest point for each ray. (cap_n_rays).
        far (Tensor): A vector containing furthest point for each ray. (cap_n_rays).
        cap_n_samples (int): The number of sampled points for each ray.
        lin_disp (bool): True for sample linearly in inverse depth rather than in depth (used for some datasets).
        perturb (bool): True for stratified sampling. False for uniform sampling.

    Returns:
        Tensor, samples where j-th component of the i-th row is the j-th sampled position along the i-th ray.
    """
    # The number of rays
    cap_n_rays = near.shape[0]

    # Uniform samples along rays
    t_vals = md.numpy.linspace(0.0, 1.0, num=cap_cap_n_samples)
    if not lin_disp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    expand_op = md.ops.BroadcastTo((cap_n_rays, cap_cap_n_samples))
    z_vals = expand_op(z_vals)

    if perturb:
        # Get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        cat_op = md.ops.Concat(-1)
        upper = cat_op([mids, z_vals[..., -1:]])
        lower = cat_op([z_vals[..., :1], mids])
        # Stratified samples in those intervals
        t_rand = md.numpy.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    return z_vals


def sample_pdf(bins, weights, cap_cap_n_samples, det=False):
    """
    Sample pdf function.

    Args:
        bins (int): The number of bins for pdf.
        weights (Tensor): The estimated weights.
        cap_cap_n_samples (int): The number of points to be sampled.
        det (bool, optional): Deterministic run or not. Default: False.

    Returns:
        Tensor, sampled pdf tensor.
    """
    weights = weights + 1e-5
    pdf = weights / md.numpy.sum(weights, -1, keepdims=True)
    cdf = md.numpy.cumsum(pdf, -1)
    cdf = md.ops.Concat(-1)([md.numpy.zeros_like(cdf[..., :1]), cdf])

    # Take uniform samples
    temp_shape = cdf.shape[:-1]
    cap_cap_n_samples_new = cap_cap_n_samples
    temp_shape_new = list(temp_shape) + [cap_cap_n_samples_new]
    if det:
        u = md.numpy.linspace(0.0, 1.0, num=cap_cap_n_samples)
        expand_op = md.ops.BroadcastTo(temp_shape_new)
        u = expand_op(u)
    else:
        u = md.numpy.rand(temp_shape_new)

    # Invert CDF
    indexes = nd_searchsorted(cdf, u)

    below = md.numpy.maximum(md.numpy.zeros_like(indexes - 1), indexes - 1)
    above = md.numpy.minimum((cdf.shape[-1] - 1) * md.numpy.ones_like(indexes), indexes)
    indexes_g = md.ops.Stack(axis=-1)([below, above])

    matched_shape = (indexes_g.shape[0], indexes_g.shape[1], cdf.shape[-1])
    gather_op = md.ops.GatherD()
    unsqueeze_op = md.ops.ExpandDims()
    expand_op = md.ops.BroadcastTo(matched_shape)
    cdf_g = gather_op(expand_op(unsqueeze_op(cdf, 1)), 2, indexes_g)
    bins_g = gather_op(expand_op(unsqueeze_op(bins, 1)), 2, indexes_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = md.numpy.where(denom < 1e-5, md.numpy.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def nd_searchsorted(cdf, u):
    """N-dim searchsorted.

    Args:
        cdf (Tensor): The cdf sampling weights.
        u (Tensor): The interval tensors.

    Returns:
        Tensor, index after searchsorted ops.
    """
    spatial_shape = cdf.shape[:-1]
    last_dim_cdf, last_dim_u = cdf.shape[-1], u.shape[-1]
    cdf_, u_ = cdf.view(-1, last_dim_cdf), u.view(-1, last_dim_u)
    indexes_ls = []

    for i in range(cdf_.shape[0]):
        indexes_ls.append(cdf_[i].searchsorted(u_[i], side="right"))
    indexes = md.ops.Stack(axis=0)(indexes_ls)
    indexes = indexes.view(*spatial_shape, last_dim_u)
    return indexes
