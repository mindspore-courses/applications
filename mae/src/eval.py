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
"""Evaluation with the test dataset."""

from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from model.vit import FineTuneVit
from process_datasets.dataset import get_dataset
from utils.logger import get_logger
from utils.eval_engine import get_eval_engine
from config.config import mae_eval_config


def main(args):
    # Initialize the environment
    local_rank = 0
    device_num = 1
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)
    args.logger.info("model config: {}".format(args))

    # Get the validation set
    eval_dataset = get_dataset(args, split="Eval")
    per_step_size = eval_dataset.get_dataset_size()
    if args.per_step_size:
        per_step_size = args.per_step_size
    args.logger.info("Create eval dataset finish, data size:{}".format(per_step_size))

    # Instantiated models
    net = FineTuneVit(batch_size=args.batch_size, patch_size=args.patch_size,
                      image_size=args.image_size, dropout=args.dropout,
                      num_classes=args.num_classes, encoder_layers=args.encoder_layers,
                      encoder_num_heads=args.encoder_num_heads, encoder_dim=args.encoder_dim,
                      mlp_ratio=args.mlp_ratio, drop_path=args.drop_path,
                      channels=args.channels)
    eval_engine = get_eval_engine(args.eval_engine, net, eval_dataset, args)

    # Load from validation checkpoint
    if args.use_ckpt:
        params_dict = load_checkpoint(args.use_ckpt)
        load_param_into_net(net, params_dict)

    # Define the model and start training
    model = Model(net, metrics=eval_engine.metric,
                  eval_network=eval_engine.eval_network)

    eval_engine.set_model(model)
    eval_engine.compile()
    eval_engine.eval()
    output = eval_engine.get_result()
    args.logger.info('accuracy={:.6f}'.format(float(output)))


if __name__ == "__main__":
    main(mae_eval_config)
