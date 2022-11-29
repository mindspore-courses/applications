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
""" Evaluate Pix2Pix Model."""

import os

from mindspore import context
from mindspore import Tensor, nn
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

from process_datasets.dataset import create_val_dataset, Pix2PixDatasetVal
from models.pix2pix import get_generator, get_discriminator, Pix2Pix
from models.loss import LossD, WithLossCellD, LossG, WithLossCellG, TrainOneStepCell
from pix2pix_utils.tools import save_image, get_lr
from config.pix2pix_config import parse_args


def pix2pix_eval(arg):
    """ Evaluate Pix2Pix Model."""
    context.set_context(mode=context.GRAPH_MODE, device_target=arg.device_target, device_id=arg.device_id)

    # Preprocess the data for evaluating
    dataset_val = Pix2PixDatasetVal(root_dir=arg.val_data_dir)
    ds_val = create_val_dataset(dataset_val, arg.val_pic_size)
    print("ds:", ds_val.get_dataset_size())
    print("ds:", ds_val.get_col_names())
    print("ds.shape:", ds_val.output_shapes())

    net_generator = get_generator(arg)
    net_discriminator = get_discriminator(arg)

    pix2pix = Pix2Pix(generator=net_generator, discriminator=net_discriminator)

    d_loss_fn = LossD(arg)
    g_loss_fn = LossG(arg)
    d_loss_net = WithLossCellD(backbone=pix2pix, loss_fn=d_loss_fn)
    g_loss_net = WithLossCellG(backbone=pix2pix, loss_fn=g_loss_fn)

    d_opt = nn.Adam(pix2pix.net_discriminator.trainable_params(), learning_rate=get_lr(arg),
                    beta1=arg.beta1, beta2=arg.beta2, loss_scale=1)
    g_opt = nn.Adam(pix2pix.net_generator.trainable_params(), learning_rate=get_lr(arg),
                    beta1=arg.beta1, beta2=arg.beta2, loss_scale=1)

    train_net = TrainOneStepCell(loss_netd=d_loss_net, loss_netg=g_loss_net, optimizerd=d_opt, optimizerg=g_opt, sens=1)
    train_net.set_train()

    # Evaluating loop
    ckpt_url = arg.ckpt
    print("CKPT:", ckpt_url)
    param_g = load_checkpoint(ckpt_url)
    load_param_into_net(net_generator, param_g)

    if not os.path.isdir(arg.predict_dir):
        os.makedirs(arg.predict_dir)

    data_loader_val = ds_val.create_dict_iterator(output_numpy=True, num_epochs=arg.epoch_num)
    print("=======Starting evaluating Loop=======")
    for i, data in enumerate(data_loader_val):
        input_image = Tensor(data["input_images"])
        fake_image = net_generator(input_image)
        save_image(fake_image, arg.predict_dir + str(i + 1))
        print("=======image", i + 1, "saved success=======")

if __name__ == '__main__':
    pix2pix_eval(parse_args())
