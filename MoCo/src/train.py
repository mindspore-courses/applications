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
""" MoCo train script."""

import time
import mindspore


def train(net, train_data, epoch, args, lr):
    """
    MoCo train

    Args:
        net : MoCo model
        train_data : train_data
        epoch: epoch
        args: train args
        lr: learning rate
    """
    start = time.time()
    net.set_train()
    total_loss, total_num, step = 0.0, 0, 0
    steps = train_data.get_dataset_size()

    for data in train_data.create_dict_iterator():
        im1 = mindspore.Tensor(data["image1"].asnumpy())
        im2 = mindspore.Tensor(data["image2"].asnumpy())
        loss = net(im1, im2)
        total_num += args.batch_size
        total_loss += loss * args.batch_size
        if step % 25 == 0:
            print(f"Epoch: [{epoch} / {args.epochs}], "
                  f"step: [{step} / {steps}], "
                  f"loss: {total_loss / total_num},"
                  f"lr: {lr}")
        step += 1
    stop = time.time() - start
    print(f"time of one step: {stop/steps} s/step ")
    return total_loss / total_num
