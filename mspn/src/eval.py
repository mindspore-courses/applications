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
""" MSPN evaluation script. """

import argparse
import json

import mindspore
from mindspore import context
from mindspore.dataset.vision.c_transforms import Normalize
from mindspore.dataset.transforms.py_transforms import Compose
import mindspore.dataset.vision.py_transforms as py_trans
import mindspore.dataset as ds

from src.utils.mspn_utils import compute_on_dataset, evaluate
from src.model.mspn import MSPN
from src.process_datasets.coco import COCODataset


def mspn_eval(args_opt):
    """MSPN eval."""
    context.set_context(mode=context.GRAPH_MODE, device_id=args_opt.device_id, device_target=args_opt.device_target)

    # Construct DataLoader
    normalize = Normalize(mean=args_opt.img_means, std=args_opt.img_stds)
    transform = Compose([normalize, py_trans.ToTensor()])
    dataset = COCODataset(data_dir=args_opt.data_url,
                          det_file_path=args_opt.det_file_url,
                          gt_file_path=args_opt.gt_file_url,
                          keypoint_num=args_opt.keypoint_num,
                          flip_pairs=args_opt.flip_pairs,
                          upper_body_ids=args_opt.upper_body_ids,
                          lower_body_ids=args_opt.lower_body_ids,
                          input_shape=args_opt.input_shape,
                          output_shape=args_opt.output_shape,
                          stage='val'
                          )
    dataloader = ds.GeneratorDataset(source=dataset,
                                     column_names=["img", "score", "center", "scale", "img_id"],
                                     shuffle=False,
                                     num_parallel_workers=args_opt.num_parallel_workers)
    dataloader = dataloader.map(operations=transform, input_columns=["img"]).batch(args_opt.batch_size)

    # Create Model
    net = MSPN(total_stage_num=args_opt.stage_num,
               output_channel_num=args_opt.keypoint_num,
               output_shape=args_opt.output_shape,
               upsample_channel_num=args_opt.upsample_channel_num,
               online_hard_key_mining=args_opt.ohkm,
               topk_keys=args_opt.topk,
               coarse_to_fine=args_opt.coarse_to_fine)

    # Load Model ckpt
    mindspore.load_param_into_net(net, mindspore.load_checkpoint(args_opt.ckpt_path))

    # Init Model
    model = mindspore.Model(net)

    # Begin to Eval
    results = compute_on_dataset(model=model,
                                 dataloader=dataloader,
                                 flip_pairs=args_opt.flip_pairs,
                                 keypoint_num=args_opt.keypoint_num,
                                 input_shape=args_opt.input_shape,
                                 output_shape=args_opt.output_shape)
    results.sort(key=lambda res: (res['image_id'], res['score']), reverse=True)
    with open(args_opt.save_path, 'w') as f:
        json.dump(results, f)

    evaluate(val_gt_path=args_opt.gt_file_url, pred_path=args_opt.save_path)


def parse_args():
    """Parse MSPN eval arguments."""
    parser = argparse.ArgumentParser(description='MSPN eval.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--device_id', type=int, default=7)
    parser.add_argument('--data_url', required=False, type=str, default="/data0/coco/coco2014", help='Data Dir Path')
    parser.add_argument('--det_file_url', required=False, type=str,
                        default="./annotation/det_json/minival_2014_det.json", help='Detection Path')
    parser.add_argument('--gt_file_url', required=False, type=str, default="./annotation/gt_json/minival_2014.json",
                        help='Label Path')
    parser.add_argument('--ckpt_path', type=str, default="./mspn.ckpt",
                        help='Location of model ckpt.')
    parser.add_argument('--save_path', type=str, default="./results.json", help='Prediction save path')
    parser.add_argument('--num_parallel_workers', type=int, default=4, help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size.')
    parser.add_argument('--stage_num', type=int, default=2, help='MSPN Stage Num.')
    parser.add_argument('--img_means', type=list, default=[0.406, 0.456, 0.485], help='Image Normalization Mean')
    parser.add_argument('--img_stds', type=list, default=[0.225, 0.224, 0.229], help='Image Normalization Std')
    parser.add_argument('--keypoint_num', type=int, default=17, help='MSPN Keypoint Num')
    parser.add_argument('--flip_pairs', type=list, default=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                                                            [15, 16]], help='MSPN Keypoint Flip Pairs')
    parser.add_argument('--upper_body_ids', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        help='MSPN Keypoint Upper Body Ids')
    parser.add_argument('--lower_body_ids', type=list, default=[11, 12, 13, 14, 15, 16],
                        help='MSPN Keypoint Lower Body Ids')
    parser.add_argument('--input_shape', type=list, default=[256, 192], help='MSPN Input Image Shape')
    parser.add_argument('--output_shape', type=list, default=[64, 48], help='MSPN Output Image Shape')
    parser.add_argument('--upsample_channel_num', type=int, default=256, help='MSPN Upsample Channel Num')
    parser.add_argument('--ohkm', type=bool, default=True, help='MSPN Online Hard Keypoint Mining')
    parser.add_argument('--topk', type=int, default=8, help='MSPN OHKM Top-k Hard Keypoint')
    parser.add_argument('--coarse_to_fine', type=bool, default=True, help='MSPN Coarse to Fine Supervision')

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    mspn_eval(parse_args())
