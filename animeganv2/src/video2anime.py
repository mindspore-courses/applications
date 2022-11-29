# Copyright 2022 Huawei Technologies Co., Ltd
# reference by: https://github.com/TachibanaYoshino/AnimeGAN/blob/master/video2anime.py
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
"""Convert real world video into anime style video."""

import argparse

import cv2
from tqdm import tqdm
from mindspore import Tensor
from mindspore import context
from mindspore import float32 as dtype
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.model import Model

from models.generator import Generator
from animeganv2_utils.adjust_brightness import adjust_brightness_from_src_to_dst
from animeganv2_utils.pre_process import preprocessing, convert_image, inverse_image


def parse_args():
    """Argument parsing"""
    parser = argparse.ArgumentParser(description='video2anime')
    parser.add_argument('--device_target', default='GPU', choices=['CPU', 'GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--video_ckpt_file_name', default='../checkpoints/Hayao/netG_30.ckpt',
                        type=str)
    parser.add_argument('--video_input', default='../video/test.mp4', type=str)
    parser.add_argument('--video_output', default='../video/output.mp4', type=str)
    parser.add_argument('--output_format', default='mp4v', type=str)
    parser.add_argument('--img_size', default=[256, 256], type=list, help='The size of image: H and W')
    return parser.parse_args()


def cvt2anime_video():
    """
    Convert the video to anime style.
    output_format: 4-letter code that specify codec to use for specific video type.
    e.g. for mp4 support use "H264", "MP4V", or "X264".
    """
    net = Generator()
    param_dict = load_checkpoint(args.video_ckpt_file_name)

    # load video
    vid = cv2.VideoCapture(args.video_input)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*args.output_format)

    load_param_into_net(net, param_dict)
    model = Model(net)

    # determine output width and height
    ret, img = vid.read()
    if img is None:
        print('Error! Failed to determine frame size: frame empty.')
        return
    img = preprocessing(img, args.img_size)
    height, width = img.shape[:2]
    out = cv2.VideoWriter(args.video_output, codec, fps, (width, height))
    pbar = tqdm(total=total)
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while ret:
        ret, frame = vid.read()
        if frame is None:
            print('Warning: got empty frame.')
            continue
        img = convert_image(frame, args.img_size)
        img = Tensor(img, dtype=dtype)
        fake_img = model.predict(img).asnumpy()
        fake_img = inverse_image(fake_img)
        fake_img = adjust_brightness_from_src_to_dst(fake_img, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        out.write(cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB))
        pbar.update(1)

    # Close the resource.
    pbar.close()
    vid.release()


if __name__ == '__main__':
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    cvt2anime_video()
