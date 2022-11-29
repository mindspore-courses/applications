"""smpl utils"""
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
from typing import NewType, Union, Optional
from dataclasses import dataclass, fields
from mindspore import Tensor
import mindspore as ms
import numpy as np


class Struct():
    """Struct object"""
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

Array = NewType("Array", np.ndarray)

def to_np(array, dtype=np.float32):
    """to np"""
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def to_tensor(array: Union[Array, Tensor], dtype=ms.float32) -> Tensor:
    """to tensor"""
    if isinstance(tuple(array), ms.Tensor):
        return array
    return Tensor(array, dtype=dtype)


@dataclass
class ModelOutput:
    """moodel output"""
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        """get"""
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        """keys"""
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        """values"""
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        """items"""
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


@dataclass
class SMPLOutput(ModelOutput):
    """smpl output"""
    betas: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None
