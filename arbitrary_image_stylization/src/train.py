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
"""Train model."""
import argparse
import os

from tqdm import tqdm
import mindspore as ms
from mindspore import nn, load_checkpoint, load_param_into_net, Tensor, context, save_checkpoint

from dataset.ais_dataset import create_dataset
from model.ais import Ais
from model.loss import TotalLoss

DEFAULT_CONTENT_WEIGHTS = {"vgg_16/conv3": 1}
DEFAULT_STYLE_WEIGHTS = {"vgg_16/conv1": 0.5e-3, "vgg_16/conv2": 0.5e-3,
                         "vgg_16/conv3": 0.5e-3, 'vgg_16/conv4': 0.5e-3}

class WithLossCell(nn.Cell):
    """
    Model with loss

    Args:
        network (nn.Cell): Arbitrary image stylization model.
        loss_fn (nn.Cell): Get sum of content loss and style loss with a vgg encoder.

    Inputs:
        -**content** (Tensor) - Tensor of shape :math:`(N, 3, H_{in}, W_{in})`.
        -**style** (Tensor) - Tensor of shape :math:`(N, 3, H_{in}, W_{in})`.

    Outputs:
        Tensor of a single value.

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> network = Ais(args.style_prediction_bottleneck)
        >>> loss = TotalLoss(3, args.content_weights, args.style_weights)
        >>> net_with_loss = WithLossCell(network, loss)
    """
    def __init__(self, network, loss_fn):
        super(WithLossCell, self).__init__()
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, content, style):
        stylized = self.network((content, style))
        loss = self.loss_fn(content, style, stylized)
        return loss

def main(args):
    """ Training procecess. """
    if args.parallel == 0:
        context.set_context(device_target=args.device_target, device_id=args.device_id)
    else:
        context.set_context(device_target=args.device_target, device_id=int(os.environ["DEVICE_ID"]))
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        init()
    # Load dataset.
    dataset = create_dataset(args)
    dataset_size = dataset.get_dataset_size()

    # Create model.
    network = Ais(args.style_prediction_bottleneck)
    loss = TotalLoss(3, args.content_weights, args.style_weights)
    load_param_into_net(loss, load_checkpoint(args.vgg_ckpt))
    load_param_into_net(network.style_predict.encoder, load_checkpoint(args.inception_v3_path))
    net_with_loss = WithLossCell(network, loss)
    opt = nn.Adam(network.trainable_params(), learning_rate=args.learning_rate)
    train_net = nn.TrainOneStepCell(net_with_loss, opt)
    train_net.set_train()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Train by steps.
    step = 0
    with tqdm(initial=step, total=args.max_step + 1) as pbar:
        for step in range(args.max_step + 1):
            if step % dataset_size == 0:
                dataloader = dataset.create_dict_iterator()
            data = next(dataloader)
            content = Tensor(data['content'])
            style = Tensor(data['style'])
            result = train_net(content, style)
            if step % args.save_checkpoint_step == 0 and step != 0:
                save_checkpoint(train_net.network.network,
                                f'{args.output}model-{int(step/args.save_checkpoint_step)}.ckpt')
            pbar.set_description(f'loss: {float(result):10.4f}')
            pbar.update(1)

def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description='arbitrary image stylization train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_step', type=int, default=3000000, help='Number of total steps.')
    parser.add_argument('--content_path', type=str, default='/data0/imagenet2012/train/', help='Path of content image.')
    parser.add_argument('--style_path', type=str, default='/data0/dtd/dtd/images/', help='Path of style image.')
    parser.add_argument('--shuffle', type=int, default=1, help='1 means True and 0 mean False')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--size', type=int, default=256, help='Image size for both content and style.')
    parser.add_argument('--content_weights', type=dict, default=DEFAULT_CONTENT_WEIGHTS,
                        help='Weights for content loss.')
    parser.add_argument('--style_weights', type=dict, default=DEFAULT_STYLE_WEIGHTS, help='Weights for style loss.')
    parser.add_argument('--style_prediction_bottleneck', type=int, default=100)
    parser.add_argument('--vgg_ckpt', type=str, default='./ckpt/vgg.ckpt', help='Path of vgg checkpoint.')
    parser.add_argument('--inception_v3_path', type=str, default='./ckpt/inception_v3.ckpt',
                        help='Path of Inception_V3 checkpoint.')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--output', type=str, default='./result/', help='Directory to save checkpoint.')
    parser.add_argument('--save_checkpoint_step', type=int, default=10000, help='Interval step of saving checkpoint.')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--parallel', type=int, default=0, help='0--training on single card,1--parallel training.')
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
