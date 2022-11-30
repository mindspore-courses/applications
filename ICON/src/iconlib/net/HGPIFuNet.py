"""HGPIFuNet module"""
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
import mindspore as ms
from mindspore import nn, ops, Tensor
from termcolor import colored
import torch

from iconlib.net.BasePIFuNet import BasePIFuNet
from iconlib.dataset.mesh_util import SMPLX, cal_sdf_batch, feat_select
from iconlib.net.MLP import MLP
from iconlib.net.HGFilters import HGFilter
from iconlib.net.NormalNet import NormalNet


class HGPIFuNet(BasePIFuNet):
    """
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    """

    def __init__(self, cfg, projection_mode="orthogonal", error_term=nn.MSELoss()):
        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode, error_term=error_term
        )

        self.l1_loss = nn.SmoothL1Loss()
        self.opt = cfg.net
        self.root = cfg.root
        self.overfit = cfg.overfit

        channels_if = self.opt.mlp_dim

        self.use_filter = self.opt.use_filter
        self.prior_type = self.opt.prior_type
        self.smpl_feats = self.opt.smpl_feats

        self.smpl_dim = self.opt.smpl_dim
        self.voxel_dim = self.opt.voxel_dim
        self.hourglass_dim = self.opt.hourglass_dim
        self.sdf_clip = cfg.sdf_clip / 100.0

        self.in_geo = [item[0] for item in self.opt.in_geo]
        self.in_nml = [item[0] for item in self.opt.in_nml]

        self.in_geo_dim = sum([item[1] for item in self.opt.in_geo])
        self.in_nml_dim = sum([item[1] for item in self.opt.in_nml])

        self.in_total = self.in_geo + self.in_nml
        self.smpl_feat_dict = None
        self.smplx_data = SMPLX()

        if "image" in self.in_geo:
            self.channels_filter = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, 8]]
        else:
            self.channels_filter = [[0, 1, 2], [3, 4, 5]]

        channels_if[0] = (
            self.hourglass_dim if self.use_filter else len(self.channels_filter[0])
        )

        if self.prior_type == "icon" and "vis" not in self.smpl_feats:
            if self.use_filter:
                channels_if[0] += self.hourglass_dim
            else:
                channels_if[0] += len(self.channels_filter[0])

        channels_if[0] += self.smpl_dim

        self.icon_keys = ["smpl_verts", "smpl_faces", "smpl_vis", "smpl_cmap"]
        self.pamir_keys = ["voxel_verts", "voxel_faces", "pad_v_num", "pad_f_num"]

        self.if_regressor = MLP(
            filter_channels=channels_if,
            name="if",
            res_layers=self.opt.res_layers,
            norm=self.opt.norm_mlp,
            last_op=nn.Sigmoid() if not cfg.test_mode else None,
        )

        # network
        if self.use_filter:
            if self.opt.gtype == "HGPIFuNet":
                self.f_filter = HGFilter(
                    self.opt, self.opt.num_stack, len(self.channels_filter[0])
                )
            else:
                print(colored(f"Backbone {self.opt.gtype} is unimplemented", "green"))

        summary_log = (
            f"{self.prior_type.upper()}:\n"
            + f"w/ Global Image Encoder: {self.use_filter}\n"
            + f"Image Features used by MLP: {self.in_geo}\n"
        )

        if self.prior_type == "icon":
            summary_log += f"Geometry Features used by MLP: {self.smpl_feats}\n"
            summary_log += "Dim of Image Features (local): 6\n"
            summary_log += f"Dim of Geometry Features (ICON): {self.smpl_dim}\n"
        elif self.prior_type == "pamir":
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PaMIR): {self.voxel_dim}\n"
        else:
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += "Dim of Geometry Features (PIFu): 1 (z-value)\n"

        summary_log += f"Dim of MLP's first layer: {channels_if[0]}\n"

        self.normal_filter = NormalNet(cfg)

    def query(self, features, points, calibs, transforms=None, regressor=None):
        """query"""
        xyz = self.projection(points, calibs, transforms)

        xy = ops.split(xyz, axis=1, output_num=2)

        in_cube = (xyz > -1.0) & (xyz < 1.0)
        in_cube = ops.stop_gradient(
            in_cube.all(axis=1, keep_dims=True).astype(ms.float32)
        )

        preds_list = []

        if self.prior_type == "icon":

            # smpl_verts [B, N_vert, 3]
            # smpl_faces [B, N_face, 3]
            # points [B, 3, N]

            smpl_sdf, smpl_norm, smpl_cmap, smpl_vis = cal_sdf_batch(
                torch.Tensor(self.smpl_feat_dict["smpl_verts"].asnumpy()),
                torch.Tensor(self.smpl_feat_dict["smpl_faces"].asnumpy()),
                torch.Tensor(self.smpl_feat_dict["smpl_cmap"].asnumpy()),
                torch.Tensor(self.smpl_feat_dict["smpl_vis"].asnumpy()),
                torch.Tensor(xyz.asnumpy()).permute(0, 2, 1).contiguous(),
            )

            smpl_sdf = Tensor(smpl_sdf.cpu().numpy())
            smpl_norm = Tensor(smpl_norm.cpu().numpy())
            smpl_cmap = Tensor(smpl_cmap.cpu().numpy())
            smpl_vis = Tensor(smpl_vis.cpu().numpy())

            # smpl_sdf [B, N, 1]
            # smpl_norm [B, N, 3]
            # smpl_cmap [B, N, 3]
            # smpl_vis [B, N, 1]

            # set ourlier point features as uniform values
            smpl_outlier = ops.ge(ops.abs(smpl_sdf), self.sdf_clip)
            smpl_sdf[smpl_outlier] = ops.Sign()(smpl_sdf[smpl_outlier])

            feat_lst = [smpl_sdf]
            if "cmap" in self.smpl_feats:
                smpl_cmap[smpl_outlier.repeat((1, 1, 3))] = smpl_sdf[
                    smpl_outlier
                ].repeat((1, 1, 3))
                feat_lst.append(smpl_cmap)
            if "norm" in self.smpl_feats:
                feat_lst.append(smpl_norm)
            if "vis" in self.smpl_feats:
                feat_lst.append(smpl_vis)

            smpl_feat = ops.transpose(ops.concat(feat_lst, axis=2), (0, 2, 1))
            vol_feats = features

        for im_feat in zip(features, vol_feats):

            # [B, Feat_i + z, N]
            # normal feature choice by smpl_vis
            if self.prior_type == "icon":
                if "vis" in self.smpl_feats:
                    point_local_feat = feat_select(
                        self.index(im_feat, xy), smpl_feat[:, [-1], :]
                    )

                    point_feat_list = [point_local_feat, smpl_feat[:, :-1, :]]
                else:
                    point_local_feat = self.index(im_feat, xy)
                    point_feat_list = [point_local_feat, smpl_feat[:, :, :]]

            point_feat = ops.concat(point_feat_list, 1)

            # out of image plane is always set to 0
            preds = regressor(point_feat)
            preds = in_cube * preds

            preds_list.append(preds)

        return preds_list

    def get_normal(self, in_tensor_dict):
        """get normal"""
        # insert normal features
        if not self.overfit:
            # print(colored("infer normal","blue"))
            feat_lst = []
            if "image" in self.in_geo:
                feat_lst.append(in_tensor_dict["image"])  # [1, 3, 512, 512]
            if "normal_F" in self.in_geo and "normal_B" in self.in_geo:
                if (
                        "normal_F" not in in_tensor_dict.keys()
                        or "normal_B" not in in_tensor_dict.keys()
                ):
                    (nml_f, nml_b) = self.normal_filter(in_tensor_dict)
                else:
                    nml_f = in_tensor_dict["normal_F"]
                    nml_b = in_tensor_dict["normal_B"]
                feat_lst.append(nml_f)  # [1, 3, 512, 512]
                feat_lst.append(nml_b)  # [1, 3, 512, 512]
            in_filter = ops.concat(feat_lst, axis=1)

        else:
            in_filter = ops.concat([in_tensor_dict[key] for key in self.in_geo], axis=1)

        return in_filter

    def filter(self, in_tensor_dict, return_inter=False):
        """
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        """

        in_filter = self.get_normal(in_tensor_dict)

        features_g = []

        if self.prior_type == "icon":
            if self.use_filter:
                features_f = self.F_filter(
                    in_filter[:, self.channels_filter[0]]
                )  # [(B,hg_dim,128,128) * 4]
                features_b = self.F_filter(
                    in_filter[:, self.channels_filter[1]]
                )  # [(B,hg_dim,128,128) * 4]
            else:
                features_f = [in_filter[:, self.channels_filter[0]]]
                features_b = [in_filter[:, self.channels_filter[1]]]
            for idx in enumerate(len(features_f)):
                features_g.append(
                    ops.concat([features_f[idx], features_b[idx]], axis=1)
                )

        if self.prior_type == "icon":
            self.smpl_feat_dict = {k: in_tensor_dict[k] for k in self.icon_keys}

        # If it is not in training, only produce the last im_feat
        if not self.training:
            features_out = [features_g[-1]]
        else:
            features_out = features_g

        if return_inter:
            return features_out, in_filter
        return features_out
