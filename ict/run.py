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
"""One stage infer for ICT."""

import os
import argparse
import sys
from subprocess import call


def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_ckpt", type=str, default='../ICT_ckpt/ms_train/Transformer/ImageNet_best.ckpt',
                        help='The path of transformer model')
    parser.add_argument("--upsample_ckpt", type=str,
                        default='../ICT_ckpt/ms_train/Upsample/InpaintingModel_gen_best.ckpt',
                        help='The path of upsample model')
    parser.add_argument("--input_image", type=str, help='The test input image path')
    parser.add_argument("--input_mask", type=str, help='The test input mask path')
    parser.add_argument("--sample_num", type=int, default=1, help='completion results')
    parser.add_argument("--save_place", type=str, default='../save', help='Please use the absolute path')
    parser.add_argument("--test_only", action='store_true', help='ImageNet pretrained model')
    opts = parser.parse_args()

    prior_url = os.path.join(opts.save_place, "AP")
    if os.path.exists(prior_url):
        print("Please change the save path")
        sys.exit(1)
    os.chdir("./Transformer")

    stage_1_command = "python infer.py --ckpt_path " + opts.transformer_ckpt + " --image_url " + opts.input_image + " \
                            --mask_url " + opts.input_mask + " --use_ImageFolder \
                            --n_layer 35 --n_embd 1024 --n_head 8 --top_k 40 --GELU_2 --image_size 32 \
                            --save_url " + prior_url + " --condition_num " + str(opts.sample_num)

    run_cmd(stage_1_command)
    print("Finish the Stage 1 - Appearance Priors Reconstruction using Transformer")

    os.chdir("../Guided_Upsample")
    if opts.test_only:
        suffix = " --test_only"
    else:
        suffix = ""
    stage_2_command = "python infer.py --input " + opts.input_image + " \
                                    --mask " + opts.input_mask + " \
                                    --prior " + prior_url + " \
                                    --save_path " + opts.save_place + " \
                                    --ckpt_path " + opts.upsample_ckpt + " \
                                    --mode 2 --mask_type 3 \
                                    --condition_num " + str(opts.sample_num) + suffix

    run_cmd(stage_2_command)
    if opts.test_only:
        run_cmd("rm -r " + opts.save_place)
    print("Finish the Stage 2 - Guided Upsampling")
