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

"""Training"""

import mindspore as ms
import mindspore.dataset as dataset
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.nn as nn
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint)
from mindspore import load_checkpoint, load_param_into_net

from configs.dataset_config import parse_args
from configs.net_configs import get_network
from dataset.mindspore_coco import COCODataset
from loss.joints_mseloss import JointsMSELoss


class CustomWithLossCell(nn.Cell):
    """
    Customized loss cell

    Args:
        net (nn.Cell): Lite-HRNet network.
        loss (nn.Cell): Loss function cell.

    Inputs:
        -**images** (Tensor) - Input image tensors.
        -**target** (Tensor) - Target heatmap.
        -**weight** (Tensor) - Weights for every joint.

    Outputs:
        -**loss_val** (Tensor) - Joints mse loss.

    Supported Platform:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> crit = CustomWithLossCell(net, joint_mse_loss)
        >>> images = mindspore.Tensor(np.random.rand(4, 3, 256, 192), mindspore.float32)
        >>> target_heatmap = mindspore.Tensor(np.random.rand(4, 17, 64, 48), mindspore.float32)
        >>> loss_val = crit(pred_heatmap, target_heatmap)
    """

    def __init__(self,
                 net: nn.Cell,
                 loss_fn: nn.Cell):
        super(CustomWithLossCell, self).__init__()
        self.net = net
        self._loss_fn = loss_fn

    def construct(self, img, target, weight):
        """ build network """

        heatmap_pred = self.net(img)
        return self._loss_fn(heatmap_pred,
                             target,
                             weight)

def main(args):
    """
    Main training function

    Args:
        args (argparse.ArgumentParser): Main configs argparser.
    """

    model_path = args.checkpoint_path
    net_type = args.model_type
    res_type = args.image_size
    epochs = args.end_epoch
    root_dir = args.root

    #Define a training dataset
    trans = Compose([py_vision.ToTensor(),
                     py_vision.Normalize(
                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_ds = COCODataset(args, root_dir, "train2017", True, transform=trans)
    print(train_ds[0][0][0].shape)

    #Constructing network model
    net = get_network(net_type, "COCO")

    #Load ckpt if necessary
    if args.load_ckpt:
        res = str(res_type[1]) + "x" + str(res_type[0])
        if "lite" in net_type:
            model_path += "litehrnet_" + str(net_type[-2:]) + "_" + "coco" + "_" + res + ".ckpt"
        else:
            model_path += net_type + "_litehrnet_18_" + "coco" + "_" + res + ".ckpt"
        param_dict = load_checkpoint(model_path)
        load_param_into_net(net, param_dict)

    #Setting training related configs
    train_loader = dataset.GeneratorDataset(train_ds, column_names=["data", "target", "weight"])
    train_loader = train_loader.batch(args.train_batch)
    optim = nn.Adam(net.trainable_params(), learning_rate=args.base_lr)
    loss = JointsMSELoss(use_target_weight=True)
    net_with_loss = CustomWithLossCell(net, loss)
    config_ck = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_steps,
                                 keep_checkpoint_max=100)
    ckpoint = ModelCheckpoint(prefix="checkpoint",
                              directory=args.checkpoint_path,
                              config=config_ck)

    model = ms.Model(network=net_with_loss, optimizer=optim)

    #Start Training
    model.train(epochs, train_loader, callbacks=[LossMonitor(), ckpoint], dataset_sink_mode=False)

if __name__ == "__main__":
    train_args = parse_args()
    main(train_args)
