import mindspore
import argparse
import numpy as np
from mindspore import context
from src.model import RetrievalWithLoss
from src.bert import BertConfig
from src.lr_schedule import Noam
from src.callbacks import TimeMonitor
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.nn import TrainOneStepCell
from src.dataset import create_dataset
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor

def parse_args():
    """set and check parameters"""
    parser = argparse.ArgumentParser(description='train duconv')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoches for training. (default: %(default)d)')
    parser.add_argument('--task_name', type=str, default='match_kn', choices=['match', 'match_kn', 'match_kn_gene'],
                        help='task name for training')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Number of word of the longest seqence. (default: %(default)d)')
    parser.add_argument('--batch_size', type=int, default=8096,
                        help='Total token number in batch for training. (default: %(default)d)')
    parser.add_argument('--vocab_size', type=int, default=14373,
                        help='Total token number in batch for training. (default: %(default)d)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate used to train with warmup. (default: %(default)f)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay rate for L2 regularizer. (default: %(default)f)')
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='proportion warmup. (default: %(default)f)')
    parser.add_argument("--train_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--train_data_shuffle", type=str, default="true", choices=["true", "false"],
                        help="Enable train data shuffle, default is true")
    parser.add_argument("--save_checkpoint_path", type=str, default="", help="Save checkpoint path")

    args = parser.parse_args()

    return args

def run_duconv():
    """run duconv task"""
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, max_call_depth=10000)
    use_kn = True if "kn" in args.task_name else False

    config = BertConfig(seq_length=args.max_seq_length, vocab_size=args.vocab_size)
    dataset = create_dataset(args.batch_size, data_file_path=args.train_data_file_path,
                             do_shuffle=(args.train_data_shuffle.lower() == "true"), use_knowledge=use_kn)
    steps_per_epoch = dataset.get_dataset_size()
    print(steps_per_epoch)

    max_train_steps = args.epoch * steps_per_epoch
    warmup_steps = int(max_train_steps * args.warmup_proportion)
    keep_checkpoint = int(max_train_steps / 1000) + 1
    network = RetrievalWithLoss(config, use_kn)
    lr_schedule = Noam(config.hidden_size, warmup_steps, args.learning_rate)
    optimizer = Adam(network.trainable_params(), lr_schedule)
    network_one_step = TrainOneStepCell(network, optimizer)
    model = Model(network_one_step)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=keep_checkpoint)
    ckpoint_cb = ModelCheckpoint(prefix=args.task_name,
                                 directory=None if args.save_checkpoint_path == "" else args.save_checkpoint_path,
                                 config=ckpt_config)
    callbacks = [TimeMonitor(100), LossMonitor(100), ckpoint_cb]

    model.train(args.epoch, dataset, callbacks, dataset_sink_mode=False)

if __name__ == "__main__":
    run_duconv()