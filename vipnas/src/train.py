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
""" Train ViPNAS Model."""

import logging

import mindspore
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import save_checkpoint
from mindspore.communication import init, get_rank, get_group_size

from utils.loss import JointsMSELoss, CustomWithLossCell
from model.top_down import create_net
import process_dataset.vipnas_image_load as ld


mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU")
init("nccl")
mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.AUTO_PARALLEL, gradients_mean=True)

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# model settings
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

train_ds = ld.TopDownCocoDataset(
    ann_file='coco/coco2017/annotations/person_keypoints_train2017.json',
    img_prefix='coco/coco2017/train2017/',
    pipeline=[ld.LoadImageFromFile(),
              ld.TopDownRandomFlip(flip_prob=0.5),
              ld.TopDownHalfBodyTransform(num_joints_half_body=8,
                                          prob_half_body=0.3),
              ld.TopDownGetRandomScaleRotation(rot_factor=30,
                                               scale_factor=0.25),
              ld.TopDownAffine(),
              ld.ToTensor(),
              ld.NormalizeTensor(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
              ld.TopDownGenerateTarget(sigma=2),
              ld.Collect(keys=['img', 'target', 'target_weight'],
                         meta_keys=['image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
                                    'rotation', 'bbox_score', 'flip_pairs'])
              ],
    data_cfg=data_cfg,
    test_mode=False
    )

rank_id = get_rank()
rank_size = get_group_size()
dataset = ds.GeneratorDataset(train_ds, ["img", "target", "target_weight"], num_shards=rank_size, shard_id=rank_id)

train_loaders = dataset.batch(64)

network = create_net(backbone='ViPNAS_ResNet')
net_opt = nn.Adam(network.trainable_params(), learning_rate=5e-4)
loss = JointsMSELoss()

loss_net = CustomWithLossCell(network, loss)

manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**24, scale_factor=2, scale_window=1000)
train_model = nn.TrainOneStepWithLossScaleCell(loss_net, net_opt, scale_sense=manager)
train_model.set_train()

filename = 'loss1.log'
logger = logging.getLogger(filename)
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(message)s')
file_handler = logging.handlers.RotatingFileHandler(
    filename=filename, maxBytes=1*1024*1024*1024, backupCount=1, encoding='utf-8')
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)

step = 0
epochs = 210
steps = train_loaders.get_dataset_size()
for epoch in range(epochs):
    step_loss = 0
    step = 0
    for d in train_loaders.create_dict_iterator():
        step_loss += train_model(d['img'], d['target'], d['target_weight'])[0].asnumpy()
        step = step + 1
        if step % 50 == 0:
            loss = step_loss / 50
            logger.info("epoch:%s, step:%s, loss:%s", epoch, step, loss)
            step_loss = 0
    if mindspore.get_context("device_target") == 2:
        file_name = "./checkpoints" + str(epoch) + ".ckpt"
        save_checkpoint(save_obj=train_model, ckpt_file_name=file_name)
