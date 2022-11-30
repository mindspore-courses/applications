"""train utils"""
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
from mindspore import ops
import mindspore as ms

from iconlib.net.geometry import orthogonal


def query_func(opt, net_g, features, points, proj_matrix=None):
    """
        - points: size of (bz, N, 3)
        - proj_matrix: size of (bz, 4, 4)
    return: size of (bz, 1, N)
    """
    assert len(points) == 1
    samples = ops.tile(points, (opt.num_views, 1, 1))
    samples = ops.transpose(samples, (0, 2, 1))  # [bz, 3, N]

    # view specific query
    if proj_matrix is not None:
        samples = orthogonal(samples, proj_matrix)

    calib_tensor = ops.stack([ops.eye(4).astype(ms.float32)], axis=0).astype(
        samples.dtype
    )

    preds = net_g.query(
        features=features,
        points=samples,
        calibs=calib_tensor,
        regressor=net_g.if_regressor,
    )

    if isinstance(preds) is list:
        preds = preds[0]

    return preds
