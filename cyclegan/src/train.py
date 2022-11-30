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
"""General-purpose training script for image-to-image translation."""

import mindspore as ms
import mindspore.nn as nn

from cyclegan_utils.args import get_args
from cyclegan_utils.reporter import Reporter
from cyclegan_utils.tools import get_lr, ImagePool, load_ckpt
from cyclegan_utils.losses import DiscriminatorLoss, GeneratorLoss
from config.cyclegan_config import parse_args
from process_datasets.dataset import create_dataset
from models.cycle_gan import get_generator, get_discriminator, Generator, TrainOneStepG, TrainOneStepD


ms.set_seed(1)

def train(args):
    """Train function."""
    args = get_args(args)

    if args.need_profiler:
        from mindspore.profiler.profiling import Profiler
        profiler = Profiler(output_path=args.outputs_dir, is_detail=True, is_show_op_path=True)
    ds = create_dataset(args)

    g_a = get_generator(args)
    g_b = get_generator(args)
    d_a = get_discriminator(args)
    d_b = get_discriminator(args)
    if args.load_ckpt:
        load_ckpt(args, g_a, g_b, d_a, d_b)
    imgae_pool_a = ImagePool(args.pool_size)
    imgae_pool_b = ImagePool(args.pool_size)
    generator = Generator(g_a, g_b, args.lambda_idt > 0)

    loss_d = DiscriminatorLoss(args, d_a, d_b)
    loss_g = GeneratorLoss(args, generator, d_a, d_b)
    optimizer_g = nn.Adam(generator.trainable_params(), get_lr(args), beta1=args.beta1)
    optimizer_d = nn.Adam(loss_d.trainable_params(), get_lr(args), beta1=args.beta1)

    net_g = TrainOneStepG(loss_g, generator, optimizer_g)
    net_d = TrainOneStepD(loss_d, optimizer_d)

    data_loader = ds.create_dict_iterator()
    reporter = Reporter(args)
    reporter.info('==========start training===============')
    for _ in range(args.max_epoch):
        reporter.epoch_start()
        for data in data_loader:
            img_a = data["image_A"]
            img_b = data["image_B"]
            res_g = net_g(img_a, img_b)
            fake_a = res_g[0]
            fake_b = res_g[1]
            res_d = net_d(img_a, img_b, imgae_pool_a.query(fake_a), imgae_pool_b.query(fake_b))
            reporter.step_end(res_g, res_d)
            reporter.visualizer(img_a, img_b, fake_a, fake_b)
        reporter.epoch_end(net_g)
        if args.need_profiler:
            profiler.analyse()
            break
    reporter.info('==========end training===============')

if __name__ == "__main__":
    train(parse_args("train"))
