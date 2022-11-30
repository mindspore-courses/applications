"""vertex ids"""
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

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# Joint name to vertex mapping. SMPL/SMPL-H/SMPL-X vertices that correspond to
# MSCOCO and OpenPose joints
vertex_ids = {
    "smplh": {
        "nose": 332,
        "reye": 6260,
        "leye": 2800,
        "rear": 4071,
        "lear": 583,
        "rthumb": 6191,
        "rindex": 5782,
        "rmiddle": 5905,
        "rring": 6016,
        "rpinky": 6133,
        "lthumb": 2746,
        "lindex": 2319,
        "lmiddle": 2445,
        "lring": 2556,
        "lpinky": 2673,
        "LBigToe": 3216,
        "LSmallToe": 3226,
        "LHeel": 3387,
        "RBigToe": 6617,
        "RSmallToe": 6624,
        "RHeel": 6787,
    },
    "smplx": {
        "nose": 9120,
        "reye": 9929,
        "leye": 9448,
        "rear": 616,
        "lear": 6,
        "rthumb": 8079,
        "rindex": 7669,
        "rmiddle": 7794,
        "rring": 7905,
        "rpinky": 8022,
        "lthumb": 5361,
        "lindex": 4933,
        "lmiddle": 5058,
        "lring": 5169,
        "lpinky": 5286,
        "LBigToe": 5770,
        "LSmallToe": 5780,
        "LHeel": 8846,
        "RBigToe": 8463,
        "RSmallToe": 8474,
        "RHeel": 8635,
    },
    "mano": {"thumb": 744, "index": 320, "middle": 443, "ring": 554, "pinky": 671,},
}
