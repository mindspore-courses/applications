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
Eval engine used for basic authentication.
"""

from mindspore import Tensor
import mindspore.common.dtype as mstype

from utils.metric import ClassifyCorrectWithCache, ClassifyCorrectCell, DistAccuracy


class BasicEvalEngine:
    """
    Basic authentication engine classes.
    """

    def __init__(self):
        print("======= __init__ =======")

    @property
    def metric(self):
        return None

    @property
    def eval_network(self):
        return None

    def compile(self):
        print("======= compile =======")

    def eval(self):
        print("======= eval =======")

    def set_model(self, model):
        self.model = model

    def get_result(self):
        return None


class ImageNetCacheEvalEngine(BasicEvalEngine):
    """
    Validation engine using cached ImageNet datasets.

    Args:
        net (net): MAE model used.
        eval_dataset (dataset): Eval dataset.
        args (dict): Parameters of the model.

    Returns:
        Float, accuracy of classification results.
    """

    def __init__(self, net, eval_dataset, args):
        super().__init__()
        self.dist_eval_network = ClassifyCorrectWithCache(net, eval_dataset)
        self.outputs = None
        self.args = args

    def compile(self):
        index = Tensor(0, mstype.int32)
        self.dist_eval_network.set_train(False)
        self.dist_eval_network.compile(index)

    def eval(self):
        index = Tensor(0, mstype.int32)
        output = self.dist_eval_network(index)
        output = output.asnumpy() / 50000
        self.outputs = {"acc": output}

    def get_result(self):
        """
        Get the classification accuracy value.

        Returns:
            Float, accuracy of classification.
        """
        return self.outputs["acc"]


class ImageNetEvalEngine(BasicEvalEngine):
    """
    Validation engine for ImageNet datasets without caching.

    Args:
        net (net): MAE model used.
        eval_dataset (dataset): Eval dataset.
        args (dict): Parameters of the model.

    Returns:
        Float, accuracy of classification results.
    """

    def __init__(self, net, eval_dataset, args):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.dist_eval_network = ClassifyCorrectCell(net)
        self.args = args
        self.outputs = None
        self.model = None

    @property
    def metric(self):
        """
        Get dist correct classification rate.

        Returns:
            Dict, the total number of correct predictions.
        """
        return {'acc': DistAccuracy(batch_size=self.args.batch_size, device_num=self.args.device_num)}

    @property
    def eval_network(self):
        """
        Get a customized cell for verifying accuracy.

        Returns:
            Cell, classify correct cell.
        """
        return self.dist_eval_network

    def eval(self):
        self.outputs = self.model.eval(self.eval_dataset)

    def get_result(self):
        """
        Get the classification accuracy value.

        Returns:
            Float, accuracy of classification.
        """
        return self.outputs["acc"]


def get_eval_engine(engine_name, net, eval_dataset, args):
    """
    Get verification engine according to the engine name.

    Args:
        engine_name (str): Engine name.
        net (net): The using network.
        eval_dataset (dataset): Eval dataset.
        args (dict): Parameters.

    Returns:
        Engine, engine for eval.
    """
    if engine_name == '':
        eval_engine = BasicEvalEngine()
    elif engine_name == "imagenet":
        eval_engine = ImageNetEvalEngine(net, eval_dataset, args)
    elif engine_name == "imagenet_cache":
        eval_engine = ImageNetCacheEvalEngine(net, eval_dataset, args)
    else:
        raise NotImplementedError

    return eval_engine
