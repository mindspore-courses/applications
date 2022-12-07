# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import numpy as np
from mindspore import dtype as mstype
from mindspore import Model, context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import create_dataset
from src.unet3d_model import UNet3d, UNet3d_
from src.utils import create_sliding_window, CalculateDice
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.Metrcis import metrics

if config.device_target == 'Ascend':
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False, \
                        device_id=device_id)
else:
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)

@moxing_wrapper()
def test_net(data_path, ckpt_path):
    data_dir = data_path + "/image/"
    seg_dir = data_path + "/seg/"
    eval_dataset = create_dataset(data_path=data_dir, seg_path=seg_dir, is_training=False)
    eval_data_size = eval_dataset.get_dataset_size()
    print("test dataset length is:", eval_data_size)

    if config.device_target == 'Ascend':
        network = UNet3d()
    else:
        network = UNet3d_()
    network.set_train(False)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)
    model = Model(network)
    # metric = metrics()
    results = {}
    results["dice"] = 0
    results["hd95"] = 0
    results["jc"] = 0
    results["asd"] = 0
    results["sens"] = 0

    index = 0
    total_dice = 0
    for batch in eval_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = batch["image"]
        seg = batch["seg"]
        print("current image shape is {}".format(image.shape), flush=True)
        sliding_window_list, slice_list = create_sliding_window(image, config.roi_size, config.overlap)
        image_size = (config.batch_size, config.num_classes) + image.shape[2:]
        output_image = np.zeros(image_size, np.float32)
        count_map = np.zeros(image_size, np.float32)
        importance_map = np.ones(config.roi_size, np.float32)
        for window, slice_ in zip(sliding_window_list, slice_list):
            window_image = Tensor(window, mstype.float32)
            pred_probs = model.predict(window_image)
            output_image[slice_] += pred_probs.asnumpy()
            count_map[slice_] += importance_map
        output_image = output_image / count_map
        dice, hd95, jc, asd, sens = CalculateDice(output_image, seg)

        print("The %d batch Dice: %.4f, HD95:%.4f, JC: %.4f, ASD: %.4f, Sens: %.4f"%(index, dice, hd95, jc, asd, sens))
        total_dice += dice
        results["dice"] += dice
        results["hd95"] += hd95
        results["jc"] += jc
        results["asd"] += asd
        results["sens"] += sens
        index = index + 1

    avg_dice = results["dice"] / eval_data_size
    avg_hd95 = results["hd95"] / eval_data_size
    avg_jc = results["jc"] / eval_data_size
    avg_asd = results["asd"] / eval_data_size
    avg_sens = results["sens"] / eval_data_size
    print("**********************End Eval***************************************")
    print("The average Dice: %.4f, HD95:%.4f, JC: %.4f, ASD: %.4f, Sens: %.4f" %
          (avg_dice, avg_hd95, avg_jc, avg_asd, avg_sens))

if __name__ == '__main__':
    test_net(data_path=config.data_path,
             ckpt_path=config.checkpoint_file_path)
