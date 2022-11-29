"""vertex joints selector"""
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
import mindspore.nn as nn
import mindspore as ms
import numpy as np

from .utils import to_tensor


class VertexJointSelector(nn.Cell):
    """vertex joints selector"""
    def __init__(self, vertex_ids=None, use_hands=True, use_feet_keypoints=True):
        super(VertexJointSelector, self).__init__()

        extra_joints_idxs = []

        face_keyp_idxs = np.array(
            [
                vertex_ids["nose"],
                vertex_ids["reye"],
                vertex_ids["leye"],
                vertex_ids["rear"],
                vertex_ids["lear"],
            ],
            dtype=np.int64,
        )

        extra_joints_idxs = np.concatenate([extra_joints_idxs, face_keyp_idxs])

        if use_feet_keypoints:
            feet_keyp_idxs = np.array(
                [
                    vertex_ids["LBigToe"],
                    vertex_ids["LSmallToe"],
                    vertex_ids["LHeel"],
                    vertex_ids["RBigToe"],
                    vertex_ids["RSmallToe"],
                    vertex_ids["RHeel"],
                ],
                dtype=np.int32,
            )

            extra_joints_idxs = np.concatenate([extra_joints_idxs, feet_keyp_idxs])

        if use_hands:
            self.tip_names = ["thumb", "index", "middle", "ring", "pinky"]

            tips_idxs = []
            for hand_id in ["l", "r"]:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            extra_joints_idxs = np.concatenate([extra_joints_idxs, tips_idxs])

        self.extra_joints_idxs = ms.Parameter(
            to_tensor(extra_joints_idxs, dtype=ms.int64), requires_grad=False
        )

    def construct(self, vertices, joints):
        extra_joints = ms.ops.gather(vertices, self.extra_joints_idxs, 1)
        joints = ms.ops.concat((joints, extra_joints), axis=1)

        return joints
