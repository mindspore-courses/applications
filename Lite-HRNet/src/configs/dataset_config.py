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
"""Some dataset related configs"""

import argparse

def parse_args():

    """
    Parse parameters.

    Returns:
        Parsed parameters.
    """

    parser = argparse.ArgumentParser(description='Lite-HRNet')

    #dataset
    parser.add_argument("--color_rgb", type=bool, default=True, help="Whether the input images are in RGB mode")
    parser.add_argument("--data_format", type=str, default="jpg", help="The format of input image")
    parser.add_argument("--flip", type=bool, default="True", help="Whether flip the input images")
    parser.add_argument("--num_joints_half_body", type=int, default=8, help="number of joints in a half body")
    parser.add_argument("--prob_half_body", type=float, default=0.3, help="Probability of doing half-body transform")
    parser.add_argument("--rot_fatcor", type=float, default=45.0, help="Rotation angle for input images")
    parser.add_argument("--scale_factor", type=float, default=0.35, help="Scaling factor for input images")
    parser.add_argument("--select_data", type=bool, default=False, help="Whether select input images with  metric")

    #model
    parser.add_argument("--target_type", type=str, default="gaussian", choices=("gaussian", "linear"),
                        help="the type of target heatmap")
    parser.add_argument("--image_size", type=list, default=[192, 256], choices=([192, 256], [288, 384]),
                        help="input image size")
    parser.add_argument("--heatmap_size", type=list, default=[48, 64], choices=([48, 64], [64, 96]),
                        help="heatmap size")
    parser.add_argument("--sigma", type=float, default=2,
                        help="sigma of the gaussian distribution that generate the target heatmap")
    parser.add_argument('--model_type', default='lite_18', type=str, help="The path of bounding box file")

    #eval
    parser.add_argument("--coco_bbox_file", type=str,
                        default='data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
                        help="The path of bounding box file")
    parser.add_argument("--image_thre", type=float, default=0.0,
                        help="The threshold to decide whether the detection bbox is trustable")
    parser.add_argument("--in_vis_thre", type=float, default=0.2,
                        help="The threshold to decide whether a joint is visible")
    parser.add_argument("--oks_thre", type=float, default=0.9, help="The threshold in computing oks nms")
    parser.add_argument("--use_gt_bbox", type=bool, default=True,
                        help="Whether to use ground truth bounding box instead of bounding box from detection result")
    parser.add_argument("--soft_nms", type=bool, default=False, help="Whether to compute soft nms instead of nms")
    parser.add_argument('--checkpoint_path', default='./ckpt/final_model/', type=str, help="The path of checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results/", help="Where to store result files")
    parser.add_argument("--root", type=str, default="./", help="Root directory")

    #infer
    parser.add_argument("--infer_data_root", default="./infer_data", type=str, help="The path of inferred images")
    parser.add_argument("--out_data_root", default="./out_data", type=str, help="Where to store output images")

    #train
    parser.add_argument('--base_lr', default=2e-3, type=float, help="Beginning learning rate")
    parser.add_argument('--end_epoch', default=200, type=int, help="Training epoches")
    parser.add_argument('--train_batch', default=32, type=int, help="Training batch size")
    parser.add_argument('--save_checkpoint_path', default='./ckpt/', type=str,
                        help="The path to save ckpts in training")
    parser.add_argument("--save_checkpoint_steps", default=500, type=str,
                        help="The interval between saved checkpoints")
    parser.add_argument('--load_ckpt', default=False, type=bool, help="Whether load previous checkpoints")

    return parser.parse_args()
