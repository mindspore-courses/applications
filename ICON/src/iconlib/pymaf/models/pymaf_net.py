"""Pymaf network"""
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
from mindspore.common.initializer import initializer, XavierUniform
from mindspore import Tensor, Parameter
import mindspore as ms
import mindspore.ops as ops
from mindspore.ops.function import broadcast_to
import numpy as np

from iconlib.pymaf.utils.geometry import (
    rot6d_to_rotmat,
    projection,
    rotation_matrix_to_angle_axis,
)
from iconlib.common.config import cfg
from .smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, H36M_TO_J14
from .hmr import ResnetBackbone
from .maf_extractor import MAFExtractor
from .res_module import IUVPredictLayer

BN_MOMENTUM = 0.1


class Regressor(nn.Cell):
    """regressor"""
    def __init__(self, feat_dim, smpl_mean_params):
        super().__init__()

        npose = 24 * 6

        self.fc1 = nn.Dense(feat_dim + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Dense(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Dense(1024, npose)
        self.decshape = nn.Dense(1024, 10)
        self.deccam = nn.Dense(1024, 3)
        self.decpose.weight = initializer(
            XavierUniform(gain=0.01), self.decpose.weight.shape
        )
        self.decshape.weight = initializer(
            XavierUniform(gain=0.01), self.decshape.weight.shape
        )
        self.deccam.weight = initializer(
            XavierUniform(gain=0.01), self.deccam.weight.shape
        )

        self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=64, create_transl=False)

        mean_params = np.load(smpl_mean_params)
        expand_dims = ops.ExpandDims()
        init_pose = expand_dims(Tensor.from_numpy(mean_params["pose"][:]), 0)
        init_shape = expand_dims(
            Tensor.from_numpy(mean_params["shape"][:].astype("float32")), 0
        )
        init_cam = expand_dims(Tensor.from_numpy(mean_params["cam"]), 0)

        self.init_pose = Parameter(init_pose, requires_grad=False)
        self.init_shape = Parameter(init_shape, requires_grad=False)
        self.init_cam = Parameter(init_cam, requires_grad=False)

    def construct(
            self,
            x,
            init_pose=None,
            init_shape=None,
            init_cam=None,
            n_iter=1,
            j_regressor=None,
    ):
        """construct"""
        batch_size = x.shape[0]
        expand_dims = ops.ExpandDims()

        if init_pose is None:
            init_pose = broadcast_to(self.init_pose, (batch_size, -1))
        if init_shape is None:
            init_shape = broadcast_to(self.init_shape, (batch_size, -1))
        if init_cam is None:
            init_cam = broadcast_to(self.init_cam, (batch_size, -1))

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        for i in range(n_iter):
            if i < 0:
                continue
            x_c = ops.concat([x, pred_pose, pred_shape, pred_cam], axis=1)
            x_c = self.fc1(x_c)
            x_c = self.drop1(x_c)
            x_c = self.fc2(x_c)
            x_c = self.drop2(x_c)
            pred_pose = self.decpose(x_c) + pred_pose
            pred_shape = self.decshape(x_c) + pred_shape
            pred_cam = self.deccam(x_c) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        vertices, joints, smpl_joints = self.smpl.construct(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=expand_dims(pred_rotmat[:, 0], 1),
            pose2rot=False,
        )

        # pred_vertices = Tensor(np.load("pred_vertices.npy"))
        pred_vertices = vertices
        pred_joints = joints
        pred_smpl_joints = smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(
            -1, 72
        )

        if j_regressor is not None:
            pred_joints = nn.MatMul(j_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].copy()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            "theta": ops.concat([pred_cam, pred_shape, pose], axis=1),
            "verts": pred_vertices,
            "kp_2d": pred_keypoints_2d,
            "kp_3d": pred_joints,
            "smpl_kp_3d": pred_smpl_joints,
            "rotmat": pred_rotmat,
            "pred_cam": pred_cam,
            "pred_shape": pred_shape,
            "pred_pose": pred_pose,
        }
        return output

    def construct_init(
            self,
            x_p,
            init_pose=None,
            init_shape=None,
            init_cam=None,
            j_regressor=None,
    ):
        """construct_init"""
        batch_size = x_p.shape[0]

        expand_dims = ops.ExpandDims()

        if init_pose is None:
            init_pose = broadcast_to(self.init_pose, (batch_size, -1))
        if init_shape is None:
            init_shape = broadcast_to(self.init_shape, (batch_size, -1))
        if init_cam is None:
            init_cam = broadcast_to(self.init_cam, (batch_size, -1))

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        vertices, joints, smpl_joints = self.smpl.construct(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=expand_dims(pred_rotmat[:, 0], 1),
            pose2rot=False,
        )

        pred_vertices = vertices
        pred_joints = joints
        pred_smpl_joints = smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(
            -1, 72
        )

        if j_regressor is not None:
            pred_joints = nn.MatMul(j_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].copy()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            "theta": ops.concat((pred_cam, pred_shape, pose), axis=1),
            "verts": pred_vertices,
            "kp_2d": pred_keypoints_2d,
            "kp_3d": pred_joints,
            "smpl_kp_3d": pred_smpl_joints,
            "rotmat": pred_rotmat,
            "pred_cam": pred_cam,
            "pred_shape": pred_shape,
            "pred_pose": pred_pose,
        }

        return output


class PyMAF(nn.Cell):
    """ PyMAF based Deep Regressor for Human Mesh Recovery
    PyMAF: 3D Human Pose and Shape Regression with Pyramidal
    Mesh Alignment Feedback Loop, in ICCV, 2021
    """

    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS):
        super().__init__()
        self.feature_extractor = ResnetBackbone(model="res50")

        # deconv layers
        self.inplanes = self.feature_extractor.inplanes
        self.deconv_with_bias = cfg.RES_MODEL.DECONV_WITH_BIAS
        self.deconv_layers = self._make_deconv_layer(
            cfg.RES_MODEL.NUM_DECONV_LAYERS,
            cfg.RES_MODEL.NUM_DECONV_FILTERS,
            cfg.RES_MODEL.NUM_DECONV_KERNELS,
        )

        maf_tmp = []
        for _ in range(cfg.MODEL.PyMAF.N_ITER):
            maf_tmp.append(MAFExtractor())

        self.maf_extractor = nn.CellList(maf_tmp)

        ma_feat_len = self.maf_extractor[-1].d_map.shape[0] * cfg.MODEL.PyMAF.MLP_DIM[-1]

        grid_size = 21
        linspace = ops.LinSpace()
        start = Tensor(-1, ms.float32)
        end = Tensor(1, ms.float32)
        x_v, y_v = ms.ops.meshgrid(
            (linspace(start, end, grid_size), linspace(start, end, grid_size)),
            indexing="ij",
        )
        stack = ops.Stack()
        expand_dims = ops.ExpandDims()
        points_grid = expand_dims(stack([x_v.reshape(-1), y_v.reshape(-1)]), 0)
        self.points_grid = Parameter(points_grid, requires_grad=False)
        grid_feat_len = grid_size * grid_size * cfg.MODEL.PyMAF.MLP_DIM[-1]

        regressor_tmp = []
        for i in range(cfg.MODEL.PyMAF.N_ITER):
            if i == 0:
                ref_infeat_dim = grid_feat_len
            else:
                ref_infeat_dim = ma_feat_len
            regressor_tmp.append(
                Regressor(feat_dim=ref_infeat_dim, smpl_mean_params=smpl_mean_params)
            )
        self.regressor = nn.CellList(regressor_tmp)

        dp_feat_dim = 256
        self.with_uv = cfg.LOSS.POINT_REGRESSION_WEIGHTS > 0
        if cfg.MODEL.PyMAF.AUX_SUPV_ON:
            self.dp_head = IUVPredictLayer(feat_dim=dp_feat_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.CellList(
                nn.Conv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    has_bias=False,
                ),
                nn.BatchNorm2d(num_features=planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i < 0:
                continue
            layers.append(block(self.inplanes, planes))

        return nn.CellList(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Deconv_layer used in Simple Baselines:
        Xiao et al. Simple Baselines for Human Pose Estimation and Tracking
        https://github.com/microsoft/human-pose-estimation.pytorch
        """
        assert num_layers == len(
            num_filters
        ), "ERROR: num_deconv_layers is different len(num_deconv_filters)"
        assert num_layers == len(
            num_kernels
        ), "ERROR: num_deconv_layers is different len(num_deconv_filters)"

        def _get_deconv_cfg(deconv_kernel):
            if deconv_kernel == 4:
                padding = 1
                # output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                # output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                # output_padding = 0

            return deconv_kernel, padding

        layers = []
        for i in range(num_layers):
            kernel, padding = _get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.Conv2dTranspose(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    pad_mode="pad",
                    padding=padding,
                    has_bias=self.deconv_with_bias,
                )
            )
            layers.append(nn.BatchNorm2d(num_features=planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU())
            self.inplanes = planes

        return nn.SequentialCell(*layers)

    def construct(self, x, j_regressor=None):
        """construct"""
        batch_size = x.shape[0]

        # spatial features and global features
        s_feat, g_feat = self.feature_extractor(x)

        assert cfg.MODEL.PyMAF.N_ITER >= 0 and cfg.MODEL.PyMAF.N_ITER <= 3
        if cfg.MODEL.PyMAF.N_ITER == 1:
            deconv_blocks = [self.deconv_layers]
        elif cfg.MODEL.PyMAF.N_ITER == 2:
            deconv_blocks = [self.deconv_layers[0:6], self.deconv_layers[6:9]]
        elif cfg.MODEL.PyMAF.N_ITER == 3:
            deconv_blocks = [
                self.deconv_layers[0:3],
                self.deconv_layers[3:6],
                self.deconv_layers[6:9],
            ]

        out_list = {}

        # initial parameters
        # by generating initial mesh the beforehand: smpl_output = self.init_smpl
        smpl_output = self.regressor[0].construct_init(g_feat, j_regressor=j_regressor)

        out_list["smpl_out"] = [smpl_output]
        out_list["dp_out"] = []

        # for visulization
        vis_feat_list = [ops.stop_gradient(s_feat)]

        # parameter predictions
        for rf_i in range(cfg.MODEL.PyMAF.N_ITER):
            pred_cam = smpl_output["pred_cam"]
            pred_shape = smpl_output["pred_shape"]
            pred_pose = smpl_output["pred_pose"]

            pred_cam = ops.stop_gradient(pred_cam)
            pred_shape = ops.stop_gradient(pred_shape)
            pred_pose = ops.stop_gradient(pred_pose)

            s_feat_i = deconv_blocks[rf_i](s_feat)
            s_feat = s_feat_i
            vis_feat_list.append(ops.stop_gradient(s_feat_i))

            self.maf_extractor[rf_i].im_feat = s_feat_i
            self.maf_extractor[rf_i].cam = pred_cam

            if rf_i == 0:
                transpose = ops.Transpose()
                sample_points = transpose(
                    ops.BroadcastTo((batch_size, -1, -1))(self.points_grid), (0, 2, 1))
                ref_feature = self.maf_extractor[rf_i].sampling(sample_points)
            else:
                pred_smpl_verts = ops.stop_gradient(smpl_output["verts"])
                expand_dims = ops.ExpandDims()
                pred_smpl_verts_ds = ops.matmul(
                    expand_dims(self.maf_extractor[rf_i].d_map, 0), pred_smpl_verts
                )  # [B, 431, 3]
                ref_feature = self.maf_extractor[rf_i].construct(
                    pred_smpl_verts_ds
                )  # [B, 431 * n_feat]

            smpl_output = self.regressor[rf_i].construct(
                ref_feature,
                pred_pose,
                pred_shape,
                pred_cam,
                n_iter=1,
                j_regressor=j_regressor,
            )
            out_list["smpl_out"].append(smpl_output)

        if self.training and cfg.MODEL.PyMAF.AUX_SUPV_ON:
            iuv_out_dict = self.dp_head(s_feat)
            out_list["dp_out"].append(iuv_out_dict)

        return out_list


def pymaf_net(smpl_mean_params):
    """ Constructs an PyMAF model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PyMAF(smpl_mean_params)
    return model
