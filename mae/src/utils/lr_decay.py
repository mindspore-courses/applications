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
# ==============================================================================
"""
Learning rate decay to speed up learning algorithm.
"""


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=None):
    """
    Parameter groups for layer-wise lr decay.
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58

    Args:
        model (model): Using model.
        weight_decay (float): Weight decay rating.
        no_weight_decay_list (list): The list of no weight decay.

    Returns:
        List, param groups values.
    """
    param_group_names = {}
    param_groups = {}

    # Get the number of layers in encoder
    num_layers = len(model.encoder.encoder.blocks) + 1

    for p in model.trainable_params():

        # No attenuation: all 1D parameters and model-specific parameters
        if len(p.shape) == 1 or p.name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.

        # The case with attenuation, set the corresponding attenuation weight, default is 0.05
        else:
            g_decay = "decay"
            this_decay = weight_decay

        # Get the corresponding layer_id
        layer_id = get_layer_id_for_vit(p.name, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:

            param_group_names[group_name] = {
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "weight_decay": this_decay,
                "params": [],
            }

        # Add the data of the corresponding layer
        param_group_names[group_name]["params"].append(p.name)
        param_groups[group_name]["params"].append(p.name)

    param_groups["order_params"] = {"order_params": model.trainable_params()}
    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Return different layer id cases for different parameters.

    Args:
        name (str): The parameter name.
        num_layers (int): Number of layer.

    Returns:
        Int: number of layer.
    """
    if name in ['encoder.cls_token', 'encoder.encoder_pos_embedding']:
        return 0
    if name.startswith('encoder.stem'):
        return 0
    if name.startswith('encoder.encoder.blocks'):
        return int(name.split('.')[3]) + 1
    return num_layers
