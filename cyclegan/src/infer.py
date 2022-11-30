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
"""Cycle GAN infer."""

import os

from mindspore import Tensor

from cyclegan_utils.args import get_args
from cyclegan_utils.reporter import Reporter
from cyclegan_utils.tools import save_image, load_ckpt
from config.cyclegan_config import parse_args
from models.cycle_gan import get_generator
from process_datasets.dataset import create_dataset


def infer(args):
    """Infer function."""
    args = get_args(args)
    g_a = get_generator(args)
    g_b = get_generator(args)
    g_a.set_train(True)
    g_b.set_train(True)
    load_ckpt(args, g_a, g_b)
    imgs_out = os.path.join(args.outputs_dir, "predict")
    if not os.path.exists(imgs_out):
        os.makedirs(imgs_out)
    if not os.path.exists(os.path.join(imgs_out, "fake_a")):
        os.makedirs(os.path.join(imgs_out, "fake_a"))
    if not os.path.exists(os.path.join(imgs_out, "fake_b")):
        os.makedirs(os.path.join(imgs_out, "fake_b"))
    args.data_dir = 'testA'
    ds = create_dataset(args)
    reporter = Reporter(args)
    reporter.start_predict("A to B")
    for data in ds.create_dict_iterator(output_numpy=True):
        img_a = Tensor(data["image"])
        path_a = str(data["image_name"][0], encoding="utf-8")
        path_b = path_a[0:-4] + "_fake_b.jpg"
        fake_b = g_a(img_a)
        save_image(fake_b, os.path.join(imgs_out, "fake_b", path_b))
        save_image(img_a, os.path.join(imgs_out, "fake_b", path_a))
    reporter.info('save fake_b at %s', os.path.join(imgs_out, "fake_b", path_a))
    reporter.end_predict()
    args.data_dir = 'testB'
    ds = create_dataset(args)
    reporter.dataset_size = args.dataset_size
    reporter.start_predict("B to A")
    for data in ds.create_dict_iterator(output_numpy=True):
        img_b = Tensor(data["image"])
        path_b = str(data["image_name"][0], encoding="utf-8")
        path_a = path_b[0:-4] + "_fake_a.jpg"
        fake_a = g_b(img_b)
        save_image(fake_a, os.path.join(imgs_out, "fake_a", path_a))
        save_image(img_b, os.path.join(imgs_out, "fake_b", path_b))
    reporter.info('save fake_a at %s', os.path.join(imgs_out, "fake_a", path_b))
    reporter.end_predict()

if __name__ == "__main__":
    infer(parse_args("predict"))
    