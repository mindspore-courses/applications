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
# ======================================================================
""" verification """

import time
import numpy as np


def compute_nme(preds, target):
    """
    Calculate the evaluation metrics for each batch.

    Args:
        preds (ndarray): Predicted coordinate values.
        target (ndarray): Actual coordinate value.

    Returns:
        rnme: ndarray. NME per batch.
        rion: ndarray. ION per batch.
        ripn: ndarray. IPN per batch.
    """

    batch_size = preds.shape[0]
    landmarks_num = preds.shape[1]
    rnme = np.zeros(batch_size)
    ripn = np.zeros(batch_size)
    rion = np.zeros(batch_size)

    for i in range(batch_size):
        pts_pred, pts_gt = preds[i,], target[i,]
        if landmarks_num == 68:
            inter_ocular = np.linalg.norm(pts_gt[36,] - pts_gt[45,])
            inter_pupil = np.linalg.norm(pts_gt[36:42,] - pts_gt[42:48,])
        elif landmarks_num == 98:
            inter_ocular = np.linalg.norm(pts_gt[60,] - pts_gt[72,])
            inter_pupil = np.linalg.norm(pts_gt[96,] - pts_gt[97,])
        else:
            raise ValueError('Number of landmarks is wrong')

        rnme[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)
                         ) / (inter_ocular * landmarks_num)
        rion[i] = np.linalg.norm(pts_pred - pts_gt) / inter_ocular
        ripn[i] = np.linalg.norm(pts_pred - pts_gt) / inter_pupil

    return rnme, rion, ripn


def validate(val_dataloader, net):
    """
    Model evaluation based on datasets and networks.

    Args:
        val_dataloader (BatchDataset): Test dataset.
        net (Cell): PFLD1X network.

    Returns:
        mne, ion, ipn and cost time.
    """

    nme_list = []
    ion_list = []
    ipn_list = []
    cost_time = []
    for i in val_dataloader.create_dict_iterator():
        img = i['img']
        landmark_gt = i['landmark']
        pfld_net = net
        start_time = time.time()
        _, landmarks = pfld_net(img)
        cost_time.append(time.time() - start_time)

        landmarks = landmarks.asnumpy()

        landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)

        landmark_gt = landmark_gt.reshape(
            landmark_gt.shape[0], -1, 2).asnumpy()

    nme_tmp, ion_tmp, ipn_tmp = compute_nme(landmarks, landmark_gt)
    for nme, ion, ipn in zip(nme_tmp, ion_tmp, ipn_tmp):
        nme_list.append(nme)
        ion_list.append(ion)
        ipn_list.append(ipn)

    print('=============== PFLD1X ==============')
    print('nme:{:.4f}'.format(np.mean(nme_list)))
    print('ion:{:.4f}'.format(np.mean(ion_list)))
    print('ipn:{:.4f}'.format(np.mean(ipn_list)))
    print("inference_cost_time: {0:4f}".format(np.mean(cost_time)))
