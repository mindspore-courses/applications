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
"""Callback function"""

import os
from collections import OrderedDict

import numpy as np
from mindspore import ops, Tensor, save_checkpoint
from mindspore.train.callback import Callback

from car_utils.metric import cal_psnr

class SaveCheckpoint(Callback):
    """
    Save checkpoint file by pnsr.

    Args:
        eval_model(Cell): The validate network.
        ds_eval(GeneratorDataset): The validate dataset.
        scale(int): Downscaling rate.
        save_path(str): The path to save checkpoint.
        eval_period(int): The validate period. Default: 1
    """
    def __init__(self, eval_model, ds_eval, scale, save_path, eval_period=1):
        """init"""
        super(SaveCheckpoint, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.m_psnr = 0.
        self.eval_period = eval_period
        path = os.path.join(os.path.realpath(save_path), f"{scale}x")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        print(f"Checkpoint files will save to {path}")
        self.save_path = path
        self.scale = scale

    def epoch_end(self, run_context):
        """
        Called once after epoch end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        scale = self.scale
        psnr_list = []
        if ((cur_epoch + 1) % self.eval_period) == 0:
            print("Validating...")
            for i, data in enumerate(self.ds_eval.create_dict_iterator()):
                if i > 10:
                    break
                image = data['image']
                _, reconstructed_img = self.model(image)
                image = image.asnumpy().transpose(0, 2, 3, 1)
                orig_img = np.uint8(image * 255).squeeze()
                reconstructed_img = ops.clip_by_value(reconstructed_img, 0, 1) * 255
                reconstructed_img = reconstructed_img.asnumpy().transpose(0, 2, 3, 1)
                recon_img = np.uint8(reconstructed_img).squeeze()

                psnr = cal_psnr(orig_img[scale:-scale, scale:-scale, ...],
                                recon_img[scale:-scale, scale:-scale, ...])
                psnr_list.append(psnr)
            m_psnr = np.mean(psnr_list)
            if m_psnr > self.m_psnr:
                self.m_psnr = m_psnr
                rank_id = os.getenv('RANK_ID', '0')
                kgn_save_path = os.path.join(self.save_path, f"kgn_{rank_id}.ckpt")
                usn_save_path = os.path.join(self.save_path, f"usn_{rank_id}.ckpt")
                net = cb_params.train_network
                net.init_parameters_data()
                param_dict = OrderedDict()
                for _, param in net.parameters_and_names():
                    param_dict[param.name] = param
                param_kgn = []
                param_usn = []
                for (key, value) in param_dict.items():
                    if "net1" in key:
                        each_param = {"name": key.replace("net1.", "")}
                        param_data = Tensor(value.data.asnumpy())
                        each_param["data"] = param_data
                        param_kgn.append(each_param)
                    elif "net2" in key:
                        each_param = {"name": key.replace("net2.", "")}
                        param_data = Tensor(value.data.asnumpy())
                        each_param["data"] = param_data
                        param_usn.append(each_param)
                append_info = {}
                append_info["epoch"] = cur_epoch

                save_checkpoint(param_kgn, kgn_save_path, append_dict=append_info)
                save_checkpoint(param_usn, usn_save_path, append_dict=append_info)
                print(f"epoce {cur_epoch}, Save model..., m_psnr for 10 images: {m_psnr}")
            else:
                print(f"epoce {cur_epoch}, m_psnr for 10 images: {m_psnr}")
            print("Validating Done.")

    def end(self, run_context):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        print(f"Finish training, totally epoches: {cur_epoch}, best psnr: {self.m_psnr}")
