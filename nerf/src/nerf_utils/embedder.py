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
"""nerf embedder for positional embedding"""

import mindspore
import mindspore.ops.operations as P
from mindspore import nn

__all__ = ["Embedder"]


class Embedder(nn.Cell):
    """
    Embedder for positional embedding.

    Args:
        input_dims (int): Input dimensions.
        max_freq_pow (float): Maximum frequency pow.
        num_freqs (int): Number of frequencies.
        periodic_fns (list, optional): Periodic fns. Default: [mindspore.ops.Sin(), mindspore.ops.Cos()].
        log_sampling (bool, optional): Log sampling. Default: True.
        include_input (bool, optional): Include input or not. Default: True.

    Inputs:
        inputs (Tensor) - Input tensor.

    Outputs:
        Tensor, input concatenated with positional embeddings.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> model = Embedder(1, 1)
        >>> inputs = ms.numpy.randn(1)
        >>> outputs = model(inputs)
        [0.1384 0.4426]
    """

    def __init__(
            self,
            input_dims,
            max_freq_pow,
            num_freqs,
            periodic_fns=(mindspore.ops.Sin(), mindspore.ops.Cos()),
            log_sampling=True,
            include_input=True,
    ):
        super().__init__()

        embed_fns = []
        out_dims = 0
        if include_input:
            embed_fns.append(mindspore.ops.Identity())
            out_dims += input_dims

        if log_sampling:
            freq_bands = mindspore.Tensor(2.0)**mindspore.numpy.linspace(0.0, max_freq_pow, num=num_freqs)
        else:
            freq_bands = mindspore.numpy.linspace(2.0**0.0, 2.0**max_freq_pow, num=num_freqs)

        for _ in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(p_fn)
                out_dims += input_dims

        self.embed_fns = embed_fns
        self.out_dims = out_dims

        self.freq_bands = freq_bands

    def construct(self, inputs):
        """Embedder construct."""
        out = []
        for i, fn in enumerate(self.embed_fns):
            if i == 0:
                out.append(fn(inputs))
            else:
                out.append(fn(inputs * self.freq_bands[(i - 1) // 2]))
        return P.Concat(-1)(out)
