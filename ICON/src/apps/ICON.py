"""ICON Module"""
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
from mindspore import nn

from iconlib.net.HGPIFuNet import HGPIFuNet
# from iconlib.common.seg3d_lossless import Seg3dLossless
# from iconlib.common.train_util import query_func


# ICON network
class ICON(nn.Cell):
    """ICON main network"""
    def __init__(self, cfg):
        super(ICON, self).__init__()

        self.cfg = cfg
        self.batch_size = self.cfg.batch_size

        self.use_sdf = cfg.sdf
        self.prior_type = cfg.net.prior_type
        self.mcube_res = cfg.mcube_res
        self.clean_mesh_flag = cfg.clean_mesh

        self.net_g = HGPIFuNet(
            self.cfg,
            self.cfg.projection_mode,
            error_term=nn.SmoothL1Loss() if self.use_sdf else nn.MSELoss(),
        )

        # self.recon_engine = Seg3dLossless(
        #     query_func=query_func,
        #     b_min=[[-1.0, 1.0, -1.0]],
        #     b_max=[[1.0, -1.0, 1.0]],
        #     resolutions=self.resolutions,
        #     align_corners=True,
        #     balance_value=0.50,
        #     visualize=False,
        #     debug=False,
        #     use_cuda_impl=False,
        #     faster=True,
        # )

    def test_single(self, batch):
        """Reconstruction one image per time"""
        in_tensor_dict = {}

        for name in self.in_total:
            if name in batch.keys():
                in_tensor_dict.update({name: batch[name]})

        if self.prior_type == "icon":
            for key in self.icon_keys:
                in_tensor_dict.update({key: batch[key]})
        else:
            pass

        features, inter = self.net_g.filter(in_tensor_dict, return_inter=True)
        sdf = self.recon_engine(
            opt=self.cfg, netG=self.net_g, features=features, proj_matrix=None
        )

        verts_pr, faces_pr = self.recon_engine.export_mesh(sdf)
        # if self.clean_mesh_flag:
        #     verts_pr, faces_pr = clean_mesh(verts_pr, faces_pr)

        # convert from GT to SDF
        verts_pr -= (self.resolutions[-1] - 1) / 2.0
        verts_pr /= (self.resolutions[-1] - 1) / 2.0

        return verts_pr, faces_pr, inter
