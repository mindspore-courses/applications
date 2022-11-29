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
""" Evaluate ViPNAS Model."""

import mindspore
from mindspore import Tensor, load_param_into_net, load_checkpoint
import mindspore.ops as ops

import process_dataset.vipnas_image_load as ld
from model.top_down import create_net


mindspore.context.set_context(device_target='GPU', device_id=0)
network = create_net(backbone='ViPNAS_ResNet')

param_dict = load_checkpoint("checkpoints205.ckpt")
para = load_param_into_net(network, param_dict)

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])
data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='coco/coco2017/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
)
d_s = ld.TopDownCocoDataset(
    ann_file='coco/coco2017/annotations/person_keypoints_val2017.json',
    img_prefix='coco/coco2017/val2017/',
    pipeline=[ld.LoadImageFromFile(),
              ld.TopDownAffine(),
              ld.ToTensor(),
              ld.NormalizeTensor(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
              ld.Collect(keys=['img'],
                         meta_keys=['image_file', 'center', 'scale',
                                    'rotation', 'bbox_score', 'flip_pairs'])
              ],
    data_cfg=data_cfg,
    test_mode=True
    )

output = []
expand_dims = ops.ExpandDims()
op = ops.Concat()
i = 0
while i < len(d_s.db):
    if i % 32 == 0:
        if i != 0:
            ds_output = network.construct(img=img,
                                          img_metas=img_metas,
                                          return_loss=False)

            output.append(ds_output)
        img = None
        img_metas = []
    ds = d_s[i]
    img_expand = expand_dims(Tensor(ds['img'], mindspore.float32), 0)
    if img is None:
        img = img_expand
    else:
        img = op((img, img_expand))
    img_metas.append(ds['img_metas'])
    i += 1

results = d_s.evaluate(output, 'result/')
for k, v in sorted(results.items()):
    print(f'{k}:{v}')
