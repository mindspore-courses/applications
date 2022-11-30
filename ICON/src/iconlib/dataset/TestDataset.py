"""deal test data"""
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
import glob
import random
import human_det
from mindspore import Tensor
import mindspore as ms
import torch
import numpy as np

from iconlib.pymaf.models import pymaf_net
from iconlib.pymaf.core import path_config
from iconlib.pymaf.utils.imutils import process_image
from iconlib.common.render import Render
from iconlib.smplx.body_models import SMPL
from iconlib.pymaf.models.smpl import SMPL_MODEL_DIR


class TestDataset:
    """Data set deal"""
    def __init__(self, cfg):

        random.seed(1993)

        self.image_dir = cfg["image_dir"]
        self.seg_dir = cfg["seg_dir"]
        self.has_det = cfg["has_det"]
        self.hps_type = cfg["hps_type"]
        self.smpl_type = "smpl" if cfg["hps_type"] != "pixie" else "smplx"
        self.smpl_gender = "neutral"

        self.det = None
        self.det = human_det.Detection()

        keep_lst = sorted(glob.glob(f"{self.image_dir}/*"))
        img_fmts = ["jpg", "png", "jpeg", "JPG", "bmp"]
        keep_lst = [item for item in keep_lst if item.split(".")[-1] in img_fmts]

        self.subject_list = sorted(
            [item for item in keep_lst if item.split(".")[-1] in img_fmts]
        )

        self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True)
        param_dict = ms.load_checkpoint("./data/pymaf_data/pretrained_model/PyMAF.ckpt")
        ms.load_param_into_net(self.hps, param_dict)

        gpu_device = 0

        device = torch.device(f"cuda:{gpu_device}")

        self.render = Render(size=512, device=device)

        self.smpl_model = SMPL(SMPL_MODEL_DIR, batch_size=64, create_transl=False)

        self.faces = self.smpl_model.faces

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, index):

        img_path = self.subject_list[index]
        img_name = img_path.split("/")[-1].rsplit(".", 1)[0]

        if self.seg_dir is None:
            img_icon, img_hps, img_ori, img_mask, uncrop_param = process_image(
                img_path, self.det, self.hps_type, 512
            )

            expand_dims = ms.ops.ExpandDims()

            data_dict = {
                "name": img_name,
                "image": expand_dims(img_icon, 0),
                "ori_image": img_ori,
                "mask": img_mask,
                "uncrop_param": uncrop_param,
            }

        preds_dict = self.hps.construct(img_hps)

        data_dict["smpl_faces"] = ms.ops.expand_dims(
            Tensor(self.faces.astype(np.int16), ms.int64), axis=0
        )

        output = preds_dict["smpl_out"][-1]
        scale, tran_x, tran_y = output["theta"][0, :3]
        data_dict["betas"] = output["pred_shape"]
        data_dict["body_pose"] = output["rotmat"][:, 1:]
        data_dict["global_orient"] = expand_dims(output["rotmat"][:, 0], 1)
        data_dict["smpl_verts"] = output["verts"]

        tran_x = tran_x.asnumpy()
        tran_y = tran_y.asnumpy()

        data_dict["scale"] = scale
        data_dict["trans"] = Tensor([tran_x, tran_y, 0.0], dtype=ms.float32)

        return data_dict

        # start render image
        # output = preds_dict['smpl_out'][-1]
        # pred_camera = torch.Tensor(output['theta'][:, :3].asnumpy())
        # pred_vertices = torch.Tensor(output['verts'].asnumpy())
        # renderer = PyRenderer(resolution=(224, 224))
        # pred_vertices = pred_vertices[0].cpu().numpy()
        #
        # # camera_translation = torch.stack([pred_camera[:, 1], pred_camera[:, 2],
        # #                                   2 * 5000 / (224 * pred_camera[:, 0] + 1e-9)],
        # #                                  dim=-1)
        #
        # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        # save_mesh_path = None
        # img_shape = renderer(
        #     pred_vertices,
        #     img=img_np,
        #     cam=pred_camera[0].cpu().numpy(),
        #     color_type='purple',
        #     mesh_filename=save_mesh_path)
        #
        # cv2.imwrite("example1" + '_smpl.png', img_shape)
        #
        # return preds_dict

    def render_normal(self, verts, faces, deform_verts=None):
        """render normal"""
        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_simple_mesh(verts, faces, deform_verts)
        print("start render")
        return self.render.get_clean_image(cam_ids=[0, 2])
