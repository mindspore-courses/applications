import time
import os
import stat
from typing import Optional, Union, Iterable
import numpy as np

import mindspore as ms
from mindspore import save_checkpoint
from mindspore.train.callback import Callback


class ValLossMonitor(Callback):
    """
    Monitors the train loss and the validation loss, after each epoch saves the
    best checkpoint file with lowest validation loss.

    Args:
        model (ms.Model): The model to monitor.
        dataset_val (ms.dataset): The dataset that the model needs.
        num_epochs (int): The number of epochs.
        interval (int): Every how many epochs to validate and print information. Default: 1.
        eval_start_epoch (int): From which time to validate. Default: 1.
        save_best_ckpt (bool): Whether to save the checkpoint file which performs best. Default: True.
        ckpt_directory (str): The path to save checkpoint files. Default: './'.
        best_ckpt_name (str): The file name of the checkpoint file which performs best. Default: 'best.ckpt'.
        metric_name (str): The name of metric for model evaluation. Default: 'Accuracy'.
        dataset_sink_mode (bool): Whether to use the dataset sinking mode. Default: True.

    Raises:
        ValueError: If `interval` is not more than 1.

    Examples:
    import mindspore as ms
    import mindspore.nn as nn
    import mindspore.dataset as ds
    from mindvision.classification.models import lenet
    from mindvision.classification.dataset import Mnist
    
    net = lenet()
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.001, momentum=0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction='mean')
    model = ms.Model(net, loss,opt,metrics={"Accuracy":nn.Accuracy()})
    dataset_val = Mnist("./mnist", split="test", batch_size=32, resize=32, download=True)
    dataset_val = dataset_val.run()
    monitor = ValAccMonitor(model, dataset_val, num_epochs=10)
    """

    def __init__(self,
                 model: ms.Model,
                 dataset_val: ms.dataset,
                 num_epochs: int,
                 interval: int = 1,
                 eval_start_epoch: int = 1,
                 save_best_ckpt: bool = True,
                 ckpt_directory: str = "./",
                 best_ckpt_name: str = "best.ckpt",
                 dataset_sink_mode: bool = True):
        super(ValLossMonitor, self).__init__()
        self.model = model
        self.dataset_val = dataset_val
        self.num_epochs = num_epochs
        self.eval_start_epoch = eval_start_epoch
        self.save_best_ckpt = save_best_ckpt
        self.best_res = 0
        self.dataset_sink_mode = dataset_sink_mode
        self.metric_name = 'loss'
        self.interval = interval
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.best_ckpt_path = os.path.join(ckpt_directory, best_ckpt_name)

    def apply_eval(self):
        """Model evaluation, return validation accuracy."""
        return self.model.eval(self.dataset_val, dataset_sink_mode=self.dataset_sink_mode)[self.metric_name]

    def epoch_end(self, run_context):
        """
        After epoch, print train loss and val accuracy,
        save the best ckpt file with highest validation accuracy.

        Args:
            run_context (RunContext): Context of the process running.
        """
        callback_params = run_context.original_args()
        cur_epoch = callback_params.cur_epoch_num

        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            # Validation result
            res = self.apply_eval()

            print("-" * 20)
            print(f"Epoch: [{cur_epoch: 3d} / {self.num_epochs: 3d}], "
                  f"Train Loss: [{callback_params.net_outputs.asnumpy() :5.3f}], "
                  f"{self.metric_name}: {res: 5.3f}")

            def remove_ckpt_file(file_name):
                os.chmod(file_name, stat.S_IWRITE)
                os.remove(file_name)

            # Save the best ckpt file
            if res <= self.best_res:
                self.best_res = res
                if self.save_best_ckpt:
                    if os.path.exists(self.best_ckpt_path):
                        remove_ckpt_file(self.best_ckpt_path)
                    save_checkpoint(callback_params.train_network, self.best_ckpt_path)

    # pylint: disable=unused-argument
    def end(self, run_context):
        """
        Print the best validation accuracy after network training.

        Args:
            run_context (RunContext): Context of the process running.
        """
        print("=" * 80)
        print(f"End of validation the best {self.metric_name} is: {self.best_res: 5.3f}, "
              f"save the best ckpt file in {self.best_ckpt_path}", flush=True)
