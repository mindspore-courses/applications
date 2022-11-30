"""Inference module"""
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
import argparse
import os
import mindspore
from mindspore import context, Parameter, Tensor, nn
from tqdm import tqdm
from termcolor import colored
import torch

from iconlib.common.config import cfg
from iconlib.dataset.TestDataset import TestDataset
from iconlib.dataset.mesh_util import get_optim_grid_image
from apps.ICON import ICON


losses = {
    "cloth": {"weight": 5.0, "value": 0.0},
    "edge": {"weight": 100.0, "value": 0.0},
    "normal": {"weight": 0.2, "value": 0.0},
    "laplacian": {"weight": 100.0, "value": 0.0},
    "smpl": {"weight": 1.0, "value": 0.0},
    "deform": {"weight": 20.0, "value": 0.0},
    "silhouette": {"weight": 1.0, "value": 0.0},
}


# There two function is useless, because train loop is not work now.
# class WithLossCell(nn.Cell):
#     """Loss cell module"""
#     def __init__(self, net):
#         super().__init__()
#         self.net = net

#     def construct(self, in_tensor, T_mask_F, T_mask_B):
#         in_tensor["normal_F"], in_tensor["normal_B"] = self.net.normal_filter.construct(
#             in_tensor
#         )

#         diff_F_smpl = ms.ops.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
#         diff_B_smpl = ms.ops.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])
#         losses["smpl"]["value"] = (diff_F_smpl + diff_B_smpl).mean()

#         smpl_arr = ms.ops.concat([T_mask_F, T_mask_B], axis=-1)[0]
#         gt_arr = ms.ops.transpose(
#             ms.ops.concat([in_tensor["normal_F"][0], in_tensor["normal_B"][0]], axis=2),
#             (1, 2, 0),
#         )
#         gt_arr = (gt_arr + 1.0) * 0.5
#         bg_color = ms.ops.expand_dims(
#             ms.ops.expand_dims(Tensor([0.5, 0.5, 0.5]), axis=0), axis=0
#         )
#         gt_arr = Tensor(((gt_arr - bg_color).sum(axis=-1) != 0.0), ms.float32)
#         diff_S = ms.ops.abs(smpl_arr - gt_arr)
#         losses["silhouette"]["value"] = diff_S.mean()

#         # Weighted sum of the losses
#         smpl_loss = 0.0
#         for k in ["smpl", "silhouette"]:
#             smpl_loss += losses[k]["value"] * losses[k]["weight"]

#         # print("loss:")
#         # print(smpl_loss)

#         return smpl_loss


# class TrainOneStepCell(nn.Cell):
#     """Train one step cell module"""
#     def __init__(self, net, optim):
#         super().__init__()

#         self.network = net
#         self.optim = optim
#         self.grad_op = ms.ops.GradOperation(get_by_list=True)
#         self.optimizer = self.optim

#     def construct(self, in_tensor, T_mask_F, T_mask_B):
#         loss = self.network(in_tensor, T_mask_F, T_mask_B)
#         grad_fn = self.grad_op(self.network)
#         grads = grad_fn(in_tensor, T_mask_F, T_mask_B)
#         print(loss)
#         self.optimizer(grads)
#         return loss


if __name__ == "__main__":

    # set device
    context.set_context(device_target="GPU", device_id=0, mode=mindspore.PYNATIVE_MODE)

    # loading cfg file
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu", "--gpu_device", type=int, default=0)
    parser.add_argument("-colab", action="store_true")
    parser.add_argument("-loop_smpl", "--loop_smpl", type=int, default=100)
    parser.add_argument("-patience", "--patience", type=int, default=5)
    parser.add_argument("-vis_freq", "--vis_freq", type=int, default=10)
    parser.add_argument("-loop_cloth", "--loop_cloth", type=int, default=100)
    parser.add_argument("-hps_type", "--hps_type", type=str, default="pymaf")
    parser.add_argument("-export_video", action="store_true")
    parser.add_argument("-in_dir", "--in_dir", type=str, default="./examples")
    parser.add_argument("-out_dir", "--out_dir", type=str, default="./results")
    parser.add_argument("-seg_dir", "--seg_dir", type=str, default=None)
    parser.add_argument(
        "-cfg", "--config", type=str, default="./configs/icon-filter.yaml"
    )

    args = parser.parse_args()

    # cfg read and merge
    cfg.merge_from_file(args.config)
    cfg.merge_from_file("./iconlib/pymaf/configs/pymaf_config.yaml")

    cfg_show_list = [
        "test_gpus",
        [args.gpu_device],
        "mcube_res",
        256,
        "clean_mesh",
        True,
    ]

    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()

    device = torch.device(f"cuda:{args.gpu_device}")

    # load model and dataloader
    model = ICON(cfg)
    param_dict = mindspore.load_checkpoint("./data/ckpt/ICON.ckpt")
    mindspore.load_param_into_net(model, param_dict)

    dataset_param = {
        "image_dir": args.in_dir,
        "seg_dir": args.seg_dir,
        "has_det": True,  # w/ or w/o detection
        "hps_type": args.hps_type,  # pymaf/pare/pixie
    }

    dataset = TestDataset(dataset_param)

    print(colored(f"Dataset Size: {len(dataset)}", "green"))

    pbar = tqdm(dataset)

    for data in pbar:
        pbar.set_description(f"{data['name']}")

        in_tensor = {"smpl_faces": data["smpl_faces"], "image": data["image"]}

        # The optimizer and variables
        optimed_pose = Parameter(
            data["body_pose"], requires_grad=True, name="body_pose"
        )  # [1,23,3,3]
        optimed_trans = Parameter(
            data["trans"], requires_grad=True, name="trans"
        )  # [3]
        optimed_betas = Parameter(
            data["betas"], requires_grad=True, name="betas"
        )  # [1,10]
        optimed_orient = Parameter(
            data["global_orient"], requires_grad=True, name="global_orient"
        )  # [1,1,3,3]

        optimizer_smpl = nn.SGD(
            params=[optimed_pose, optimed_trans, optimed_betas, optimed_orient],
            learning_rate=1e-3,
            momentum=0.9,
        )

        # loop_smpl = tqdm(
        #     range(args.loop_smpl if cfg.net.prior_type != "pifu" else 1))

        # net_with_loss = WithLossCell(model.netG)
        # train_step = TrainOneStepCell(net_with_loss, optimizer_smpl)

        # for i in loop_smpl:
        #     smpl_out = dataset.smpl_model.construct(
        #         betas=optimed_betas,
        #         body_pose=optimed_pose,
        #         global_orient=optimed_orient,
        #         pose2rot=False,
        #     )

        #     smpl_verts = ((smpl_out.vertices) +
        #                 optimed_trans) * data["scale"]

        #     smpl_verts = torch.Tensor(smpl_verts.asnumpy())

        #     l = smpl_verts[0]

        #     l[:, 1] = l[:, 1] * -1
        #     l[:, 2] = l[:, 2] * -1

        #     smpl_verts[0] = l

        #     in_tensor["smpl_faces"] = torch.Tensor(in_tensor["smpl_faces"].asnumpy())

        #     # render optimized mesh (normal, T_normal, image [-1,1])
        #     in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
        #         smpl_verts, in_tensor["smpl_faces"]
        #     )

        #     T_mask_F, T_mask_B = dataset.render.get_silhouette_image()

        #     in_tensor["smpl_faces"] = Tensor(in_tensor["smpl_faces"].cpu().numpy())

        #     in_tensor["T_normal_F"] = Tensor(in_tensor["T_normal_F"].cpu().numpy())

        #     in_tensor["T_normal_B"] = Tensor(in_tensor["T_normal_B"].cpu().numpy())

        #     T_mask_F = Tensor(T_mask_F.cpu().numpy())
        #     T_mask_B = Tensor(T_mask_B.cpu().numpy())

        #     loss = train_step(in_tensor, T_mask_F, T_mask_B)

        smpl_out = dataset.smpl_model.construct(
            betas=optimed_betas,
            body_pose=optimed_pose,
            global_orient=optimed_orient,
            pose2rot=False,
        )

        smpl_verts = ((smpl_out.vertices) + optimed_trans) * data["scale"]

        smpl_verts = torch.Tensor(smpl_verts.asnumpy())

        l = smpl_verts[0]

        l[:, 1] = l[:, 1] * -1
        l[:, 2] = l[:, 2] * -1

        smpl_verts[0] = l

        in_tensor["smpl_faces"] = torch.Tensor(in_tensor["smpl_faces"].asnumpy())

        # render optimized mesh (normal, T_normal, image [-1,1])
        in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
            smpl_verts, in_tensor["smpl_faces"]
        )

        T_mask_F, T_mask_B = dataset.render.get_silhouette_image()

        in_tensor["smpl_faces"] = Tensor(in_tensor["smpl_faces"].cpu().numpy())

        in_tensor["T_normal_F"] = Tensor(in_tensor["T_normal_F"].cpu().numpy())

        in_tensor["T_normal_B"] = Tensor(in_tensor["T_normal_B"].cpu().numpy())

        (
            in_tensor["normal_F"],
            in_tensor["normal_B"],
        ) = model.net_g.normal_filter.construct(in_tensor)

        diff_F_smpl = mindspore.ops.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
        diff_B_smpl = mindspore.ops.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])

        in_tensor["image"] = torch.Tensor(in_tensor["image"].asnumpy()).to(device)
        in_tensor["T_normal_F"] = torch.Tensor(in_tensor["T_normal_F"].asnumpy()).to(
            device
        )
        in_tensor["normal_F"] = torch.Tensor(in_tensor["normal_F"].asnumpy()).to(device)

        in_tensor["T_normal_B"] = torch.Tensor(in_tensor["T_normal_B"].asnumpy()).to(
            device
        )
        in_tensor["normal_B"] = torch.Tensor(in_tensor["normal_B"].asnumpy()).to(device)

        # silhouette loss
        smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)[0].to(device)

        # smpl_arr = Tensor(smpl_arr.cpu().numpy())

        gt_arr = (
            torch.cat([in_tensor["normal_F"][0], in_tensor["normal_B"][0]], dim=2)
            .permute(1, 2, 0)
            .to(device)
        )

        bg_color = torch.Tensor([0.5, 0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(device)

        gt_arr = ((gt_arr - bg_color).sum(dim=-1) != 0.0).float()

        # gt_arr = ms.ops.transpose(ms.ops.concat(
        #         [in_tensor["normal_F"][0], in_tensor["normal_B"][0]], axis=2
        #     ), (1, 2, 0))
        gt_arr = ((gt_arr + 1.0) * 0.5).to(device)

        # diff_S = ms.ops.abs(smpl_arr - gt_arr)
        diff_S = torch.abs(smpl_arr - gt_arr)

        per_loop_lst = []
        per_data_lst = []

        diff_F_smpl = torch.Tensor(diff_F_smpl.asnumpy()).to(device)
        diff_B_smpl = torch.Tensor(diff_B_smpl.asnumpy()).to(device)

        per_loop_lst.extend(
            [
                in_tensor["image"],
                in_tensor["T_normal_F"],
                in_tensor["normal_F"],
                diff_F_smpl / 2.0,
                diff_S[:, :512].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
            ]
        )
        per_loop_lst.extend(
            [
                in_tensor["image"],
                in_tensor["T_normal_B"],
                in_tensor["normal_B"],
                diff_B_smpl / 2.0,
                diff_S[:, 512:].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
            ]
        )
        per_data_lst.append(
            get_optim_grid_image(per_loop_lst, None, nrow=5, type_p="smpl")
        )

        # per_loop_lst.extend(
        #             [
        #                 in_tensor["image"],
        #                 in_tensor["T_normal_F"],
        #                 in_tensor["normal_F"],
        #                 diff_F_smpl / 2.0,
        #                 ms.numpy.tile(ms.ops.expand_dims(ms.ops.expand_dims(diff_S[:, :512], axis=0), axis=0), (1, 3, 1, 1))
        #             ]
        # )

        # per_loop_lst.extend(
        #             [
        #                 in_tensor["image"],
        #                 in_tensor["T_normal_B"],
        #                 in_tensor["normal_B"],
        #                 diff_F_smpl / 2.0,
        #                 ms.numpy.tile(ms.ops.expand_dims(ms.ops.expand_dims(diff_S[:, :512], axis=0), axis=0), (1, 3, 1, 1))
        #             ]
        # )

        # visualize the optimization process
        # 1. SMPL Fitting
        # 2. Clothes Refinement

        os.makedirs(os.path.join(args.out_dir, cfg.name, "refinement"), exist_ok=True)

        # visualize the final results in self-rotation mode
        os.makedirs(os.path.join(args.out_dir, cfg.name, "vid"), exist_ok=True)

        # final results rendered as image
        # 1. Render the final fitted SMPL (xxx_smpl.png)
        # 2. Render the final reconstructed clothed human (xxx_cloth.png)
        # 3. Blend the original image with predicted cloth normal (xxx_overlap.png)

        os.makedirs(os.path.join(args.out_dir, cfg.name, "png"), exist_ok=True)

        # final reconstruction meshes
        # 1. SMPL mesh (xxx_smpl.obj)
        # 2. clohted mesh (xxx_recon.obj)
        # 3. refined clothed mesh (xxx_refine.obj)

        os.makedirs(os.path.join(args.out_dir, cfg.name, "obj"), exist_ok=True)

        per_data_lst[-1].save(
            os.path.join(args.out_dir, cfg.name, f"png/{data['name']}_smpl.png")
        )

        # verts_pr, faces_pr, _ = model.test_single(in_tensor)

        # print(verts_pr)
