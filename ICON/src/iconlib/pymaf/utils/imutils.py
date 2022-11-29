"""image utils"""
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
import io
import cv2
from mindspore.dataset import transforms, vision
from mindspore import Tensor
import mindspore as ms
import numpy as np
from PIL import Image
from rembg.bg import remove
import torch

from iconlib.pymaf.utils.streamer import aug_matrix


def process_image(img_file, det, hps_type, input_res=512, seg_path=None):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available,
    use them to get the bounding box.
    """
    [
        image_to_tensor,
        mask_to_tensor,
        image_to_pixie_tensor,
    ] = get_transformer(input_res)

    if hps_type and seg_path:
        print("hps is in")

    img_ori = load_img(img_file)

    in_height, in_width, _ = img_ori.shape
    m_p = aug_matrix(in_width, in_height, input_res * 2, input_res * 2)

    # from rectangle to square
    img_for_crop = cv2.warpAffine(
        img_ori, m_p[0:2, :], (input_res * 2, input_res * 2), flags=cv2.INTER_CUBIC
    )

    if det is not None:

        # detection for bbox
        bbox = get_bbox(img_for_crop, det)

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
    else:
        height = img_for_crop.shape[0]
        width = img_for_crop.shape[1]
        center = np.array([width // 2, height // 2])

    scale = max(height, width) / 180

    img_np = crop(
        img_for_crop, center, scale, (input_res, input_res)
    )

    buf = ms.ops.stop_gradient(io.BytesIO())
    Image.fromarray(img_np).save(buf, format="png")
    img_pil = ms.ops.stop_gradient(
        Image.open(io.BytesIO(remove(buf.getvalue()))).convert("RGBA")
    )

    # for icon
    img_rgb = Tensor(image_to_tensor(img_pil.convert("RGB"))[0])
    img_mask = Tensor(1.0) - ms.ops.squeeze(
        Tensor(mask_to_tensor(img_pil.split()[-1]), dtype=ms.float32)
        < Tensor(0.5, dtype=ms.float32),
        axis=0,
    )

    img_tensor = img_rgb * img_mask

    # for hps
    transpose = ms.ops.Transpose()
    img_hps = img_np.astype(np.float32) / 255.0

    img_hps = Tensor.from_numpy(np.ascontiguousarray(img_hps)).asnumpy()

    expand_dims = ms.ops.ExpandDims()

    img_hps = expand_dims(
        transpose(Tensor(image_to_pixie_tensor(img_hps)), (2, 0, 1)), 0
    )

    # uncrop params
    uncrop_param = {
        "center": center,
        "scale": scale,
        "ori_shape": img_ori.shape,
        "box_shape": img_np.shape,
        "crop_shape": img_for_crop.shape,
        "M": m_p,
    }

    return img_tensor, img_hps, img_ori, img_mask, uncrop_param


def get_transformer(input_res):
    """get transformer"""
    image_to_tensor = transforms.transforms.Compose(
        [
            vision.Resize(input_res),
            vision.ToTensor(),
            vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
        ]
    )

    mask_to_tensor = transforms.transforms.Compose(
        [
            vision.Resize(input_res),
            vision.ToTensor(),
            vision.Normalize(mean=[0.0], std=[1.0], is_hwc=False),
        ]
    )

    image_to_pixie_tensor = transforms.transforms.Compose([vision.Resize([224, 224])])

    return [
        image_to_tensor,
        mask_to_tensor,
        image_to_pixie_tensor,
    ]


def load_img(img_file):
    """load image"""
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if not img_file.endswith("png"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img


def get_bbox(img, det):
    """get bbox"""
    input_p = np.float32(img)
    input_p = (input_p / 255.0 - (0.5, 0.5, 0.5)) / (0.5, 0.5, 0.5)  # TO [-1.0, 1.0]
    input_p = input_p.transpose(2, 0, 1)  # TO [3 x H x W]
    bboxes, probs = det(torch.from_numpy(input_p).float().unsqueeze(0))

    probs = probs.unsqueeze(3)
    bboxes = (bboxes * probs).sum(dim=1, keepdim=True) / probs.sum(dim=1, keepdim=True)
    bbox = bboxes[0, 0, 0].cpu().numpy()

    return bbox


def get_transform(center, scale, res):
    """Generate transformation matrix."""
    h_p = 200 * scale
    t_p = np.zeros((3, 3))
    t_p[0, 0] = float(res[1]) / h_p
    t_p[1, 1] = float(res[0]) / h_p
    t_p[0, 2] = res[1] * (-float(center[0]) / h_p + 0.5)
    t_p[1, 2] = res[0] * (-float(center[1]) / h_p + 0.5)
    t_p[2, 2] = 1

    return t_p


def transform(pt_p, center, scale, res, invert=0):
    """Transform pixel location to different reference."""
    t_p = get_transform(center, scale, res)
    if invert:
        t_p = np.linalg.inv(t_p)
    new_pt = np.array([pt_p[0] - 1, pt_p[1] - 1, 1.0]).T
    new_pt = np.dot(t_p, new_pt)
    return np.around(new_pt[:2]).astype(np.int16)


def crop(img, center, scale, res):
    """Crop image according to the supplied bounding box."""

    # Upper left point
    ul_p = np.array(transform([0, 0], center, scale, res, invert=1))

    # Bottom right point
    br_p = np.array(transform(res, center, scale, res, invert=1))

    new_shape = [br_p[1] - ul_p[1], br_p[0] - ul_p[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul_p[0]), min(br_p[0], len(img[0])) - ul_p[0]
    new_y = max(0, -ul_p[1]), min(br_p[1], len(img)) - ul_p[1]

    # Range to sample from original image
    old_x = max(0, ul_p[0]), min(len(img[0]), br_p[0])
    old_y = max(0, ul_p[1]), min(len(img), br_p[1])

    new_img[new_y[0] : new_y[1], new_x[0] : new_x[1]] = img[
        old_y[0] : old_y[1], old_x[0] : old_x[1]
    ]
    if len(img.shape) == 2:
        new_img = np.array(Image.fromarray(new_img).resize(res))
    else:
        new_img = np.array(Image.fromarray(new_img.astype(np.uint8)).resize(res))

    return new_img
