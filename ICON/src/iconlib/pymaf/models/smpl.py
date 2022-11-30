"""pymaf smpl module"""
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
# This script is borrowed from https://github.com/nkolot/SPIN/blob/master/models/smpl.py
from collections import namedtuple
import mindspore as ms
from mindspore import Tensor
import numpy as np

from iconlib.smplx.body_models import SMPL as _SMPL, ModelOutput
from iconlib.smplx.libs import vertices2joints
from iconlib.pymaf.core import path_config, constants

SMPL_MEAN_PARAMS = path_config.SMPL_MEAN_PARAMS
SMPL_MODEL_DIR = path_config.SMPL_MODEL_DIR

# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        j_regressor_extra = np.load(path_config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.j_regressor_extra = ms.Parameter(
            Tensor(j_regressor_extra, dtype=ms.float32), requires_grad=False
        )
        self.joint_map = Tensor(joints, dtype=ms.int64)
        self.module_output = namedtuple(
            "ModelOutput_", ModelOutput._fields + ("smpl_joints", "joints_J19",)
        )
        self.module_output.__new__.__defaults__ = (None,) * len(self.module_output._fields)

    def construct(self, *args, **kwargs):
        # kwargs["get_skin"] = True
        smpl_output = super().construct(*args, **kwargs)
        extra_joints = vertices2joints(self.j_regressor_extra, smpl_output.vertices)
        # smpl_output.joints: [B, 45, 3]  extra_joints: [B, 9, 3]
        vertices = smpl_output.vertices
        joints = ms.ops.concat((smpl_output.joints, extra_joints), axis=1)
        smpl_joints = smpl_output.joints[:, :24]
        joints = joints[:, self.joint_map, :]  # [B, 49, 3]
        return vertices, joints, smpl_joints


def get_smpl_faces():
    """get smpl face"""
    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    return smpl.faces


def get_part_joints(smpl_joints):
    """get part joints"""
    # part_joints = torch.zeros().to(smpl_joints.device)

    one_seg_pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (3, 6),
        (9, 12),
        (9, 13),
        (9, 14),
        (12, 15),
        (13, 16),
        (14, 17),
    ]
    two_seg_pairs = [
        (1, 4),
        (2, 5),
        (4, 7),
        (5, 8),
        (16, 18),
        (17, 19),
        (18, 20),
        (19, 21),
    ]

    one_seg_pairs.extend(two_seg_pairs)

    single_joints = [(10), (11), (15), (22), (23)]

    part_joints = []

    for j_p in one_seg_pairs:
        mean = ms.ops.ReduceMean(keep_dims=True)
        new_joint = mean(smpl_joints[:, j_p], axis=1)
        part_joints.append(new_joint)

    for j_p in single_joints:
        part_joints.append(smpl_joints[:, j_p : j_p + 1])

    cat = ms.ops.Concat()
    part_joints = cat(part_joints, 1)

    return part_joints
