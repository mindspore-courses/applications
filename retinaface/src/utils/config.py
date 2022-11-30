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
"""Config for train and eval."""

global_config = {

    # resnet50 or mobilenet025
    'backbone': 'resnet50',
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'class_weight': 1.0,
    'landm_weight': 1.0,
    'batch_size': 1,
    'num_workers': 2,
    'num_anchor': 16800,
    'ngpu': 0,
    'image_size': 640,
    'match_thresh': 0.35,
    'in_channel': 32,
    'out_channel': 64,
    'target_size': 840,
    'max_size': 840,

    # opt
    'optim': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,

    # seed
    'seed': 1,

    # lr
    'epoch': 120,
    'decay1': 70,
    'decay2': 90,
    'lr_type': 'dynamic_lr',

    # for resnet50,0.001-0.0005 is better,0.00065 suggest.for mobilenet025,0.01-0.001 is better,0.005 suggest.
    'initial_lr': 0.005,
    'warmup_epoch': 5,
    'gamma': 0.1,

    # checkpoint
    'ckpt_path': './checkpoint/',
    'save_checkpoint_steps': 100,
    'keep_checkpoint_max': 3,
    'resume_net': None,

    # dataset
    'training_dataset': './../../data/widerface/train/label.txt',
    'pretrain': False,
    'pretrain_path': './data/res50_pretrain.ckpt',

    # val
    'val_model': './../../data/pretrained_model/retinafaceresnet50.ckpt',
    'val_dataset_folder': './../../data/valid/',
    'val_origin_size': False,
    'val_confidence_threshold': 0.02,
    'val_nms_threshold': 0.4,
    'val_iou_threshold': 0.5,
    'val_save_result': True,
    'val_predict_save_folder': './widerface_result',
    'val_gt_dir': './../../data/ground_truth/',

}

facealignment_config = {

    # Helen Dataset
    'img_size': 192,
    'dataset_side_data_enhance': 'False',
    'dataset_target_path': 'Helen_192_no_enhance_do_clip',
    'clip': True,

    # FaceAlignment Config
    'num_classes': 388,
    'batch_size': 4,
    'epoch_size': 1000,
    'warmup_epochs': 4,
    'lr': 0.0001,
    'momentum': 0.9,
    'weight_decay': 0.00004,
    'loss_scale': 1024,
    'save_checkpoint': True,
    'save_checkpoint_epochs': 10,
    'keep_checkpoint_max': 500,
    'save_checkpoint_path': "./checkpoint",
    'export_format': "MINDIR",
    'export_file': "FaceAlignment_2D"
}
