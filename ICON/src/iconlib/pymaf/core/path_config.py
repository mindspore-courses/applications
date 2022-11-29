"""path config"""
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
import os

# pymaf
pymaf_data_dir = os.path.join(os.path.dirname(__file__), "../../../data/pymaf_data")

SMPL_MEAN_PARAMS = os.path.join(pymaf_data_dir, "smpl_mean_params.npz")
SMPL_MODEL_DIR = os.path.join(pymaf_data_dir, "../smpl_related/models/smpl")

CUBE_PARTS_FILE = os.path.join(pymaf_data_dir, "cube_parts.npy")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(pymaf_data_dir, "J_regressor_extra.npy")
JOINT_REGRESSOR_H36M = os.path.join(pymaf_data_dir, "J_regressor_h36m.npy")
VERTEX_TEXTURE_FILE = os.path.join(pymaf_data_dir, "vertex_texture.npy")
SMPL_MEAN_PARAMS = os.path.join(pymaf_data_dir, "smpl_mean_params.npz")
SMPL_MODEL_DIR = os.path.join(pymaf_data_dir, "../smpl_related/models/smpl")
CHECKPOINT_FILE = os.path.join(
    pymaf_data_dir, "pretrained_model/PyMAF_model_checkpoint.pt"
)

# pare
pare_data_dir = os.path.join(os.path.dirname(__file__), "../../../data/pare_data")
CFG = os.path.join(pare_data_dir, "pare/checkpoints/pare_w_3dpw_config.yaml")
CKPT = os.path.join(pare_data_dir, "pare/checkpoints/pare_w_3dpw_checkpoint.ckpt")

# hybrik
hybrik_data_dir = os.path.join(os.path.dirname(__file__), "../../../data/hybrik_data")
HYBRIK_CFG = os.path.join(hybrik_data_dir, "hybrik_config.yaml")
HYBRIK_CKPT = os.path.join(hybrik_data_dir, "pretrained_w_cam.pth")
