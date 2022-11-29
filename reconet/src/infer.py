# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.lr_schedule.py
# org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" ReCoNet infer script."""

import argparse
import time

import cv2
from mindspore import context
from mindspore.train import Model

from model.reconet import load_reconet
from utils.reconet_utils import preprocess, save_infer_result, postprocess, batch_style_transfer


def infer_image(args_opt, model):
    """
    Infer for image

    Args:
        args_opt (str): infer args
        model (ReCoNet): ReCoNet model
    """

    # preprocess input image
    image = preprocess(args_opt.input_file)

    # style input image
    styled_image = model.predict(image).squeeze()

    # post process and save image to the output file
    save_infer_result((styled_image + 1) / 2, args_opt.output_file)


def init_video_cap(args_opt):
    """
    Infer for image

    Args:
        args_opt (str): infer args

    Returns:
        capture, video capture
        write, video writer
        ret, whether have next frame
        img, image frame
    """
    capture = cv2.VideoCapture(args_opt.input_file)
    ret, img = capture.read()
    height, width = img.shape[:2]
    writer = cv2.VideoWriter(args_opt.output_file, cv2.VideoWriter_fourcc(*'mp4v'), int(capture.get(cv2.CAP_PROP_FPS)),
                             (width, height))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return capture, writer, ret, img


def infer_video(args_opt, model):
    """
    Infer for video

    Args:
        args_opt (str): infer args
        model (ReCoNet): ReCoNet model
    """
    # init video capture and video writer
    cap, writer, ret, img = init_video_cap(args_opt)

    batch = [img]

    # transfer frame one by one
    while ret:
        ret, frame = cap.read()
        if frame is None:
            print('Empty frame.')
            continue
        batch.append(frame)

        if batch.__len__() == 2:
            input_batch = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in batch]
            for output_frame in batch_style_transfer(input_batch, model):
                writer.write(cv2.cvtColor(postprocess(output_frame), cv2.COLOR_RGB2BGR))
            batch = []

    if batch.__len__() != 0:
        input_batch = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in batch]
        for output_frame in batch_style_transfer(input_batch, model):
            writer.write(cv2.cvtColor(postprocess(output_frame), cv2.COLOR_RGB2BGR))

    # close video capture
    cap.release()


def reconet_infer(args_opt):
    """
    ReCoNet infer

    Args:
        args_opt (str): infer args
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Create model.
    network = load_reconet(ckpt_file=args_opt.ckpt_file)

    network.set_train(False)

    # Init the model.
    model = Model(network)

    print('Infer start in [{}] mode'.format(args_opt.infer_mode))

    start = time.perf_counter()
    if args_opt.infer_mode.lower() == 'image':
        infer_image(args_opt, model)
    else:
        infer_video(args_opt, model)
    end = time.perf_counter()
    print(f'infer done, time cost {(end - start) * 1000: .3f} ms')


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description='Reconet infer.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--input_file', required=True, default=None, help='Path of input file, can be image or video')
    parser.add_argument('--ckpt_file', required=True, type=str, default=None, help='Path of the check point file.')
    parser.add_argument('--output_file', required=True, type=str, default=None, help='Path of the output file.')
    parser.add_argument('--infer_mode', required=True, type=str, default="image", choices=["image", "video"],
                        help='Support style transfer for image and video')
    args = parser.parse_known_args()[0]
    return args

if __name__ == '__main__':
    reconet_infer(parse_args())
