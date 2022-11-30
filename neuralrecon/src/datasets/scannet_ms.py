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
"""Dataset and dataloader for MindSpore"""

import os
import pickle

from PIL import Image
import numpy as np
import cv2
import mindspore.dataset as ds

from sampler_ms import DistributedSampler


def GetDataLoader(datapath,
                  transforms,
                  nviews,
                  n_scales,
                  per_batch_size,
                  max_epoch,
                  rank,
                  group_size,
                  mode='train'):
    """
    NeuralRecon get data loader
    """
    neuralrecon_gen = ScanNetDataset(datapath, mode, transforms, nviews, n_scales)
    sampler = DistributedSampler(neuralrecon_gen, rank, group_size, shuffle=(mode == 'train')) # user defined sampling strategy
    de_dataset = ds.GeneratorDataset(neuralrecon_gen, ["items"], sampler=sampler, num_parallel_workers=4)

    if group_size > 1:
        num_parallel_workers = 8
    else:
        num_parallel_workers = 16
    if mode == 'train':
        compose_map_func = transforms
        columns = ["items"]
        de_dataset = de_dataset.map(input_columns=["items"],
                                    output_columns=columns,
                                    operations=compose_map_func,
                                    num_parallel_workers=num_parallel_workers,
                                    python_multiprocessing=True)

    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=True, num_parallel_workers=8)
    if mode == 'train':
        #de_dataset = de_dataset.repeat(1) # if use this, need an additional "for" cycle epoch times
        de_dataset = de_dataset.repeat(max_epoch)

    return de_dataset, de_dataset.get_dataset_size()


class ScanNetDataset:
    """Scannet dataset"""
    def __init__(self, datapath, mode, transforms, nviews, n_scales):
        """
        Init for scannet dataset
        """
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.transforms = transforms
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100

    def build_list(self):
        """Build list"""
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        """Read cam file"""
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        extrinsics = extrinsics.astype(np.float32)
        return intrinsics, extrinsics

    def read_img(self, filepath):
        """Read img"""
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        """Read depth"""
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def read_scene_volumes(self, data_path, scene):
        """Read scene volumes"""
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0.astype(np.float32))
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]

        imgs = []
        depth = []
        extrinsics_list = []
        intrinsics_list = []

        tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])

        for vid in meta['image_ids']:
            # load images
            imgs.append(
                self.read_img(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid))))

            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}.png'.format(vid)))
            )

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),
                                                        vid)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)

        items = {
            'imgs': imgs,
            'depth': depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'vol_origin': meta['vol_origin'].astype(np.float32),
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
        }

        if self.transforms is not None:
            items = self.transforms(items)
        return items
