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
"""MSPN Utils"""
import os

import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mindspore import Tensor

from src.utils.keypoints_transforms import flip_back


def ensure_dir(path) -> None:
    """
    Make sure directory exists

    Args:
        path (str): Dir Path

    Examples:
        >>> ensure_dir(r'./dir')
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def get_results(outputs: np.ndarray,
                centers: np.ndarray,
                scales: np.ndarray,
                keypoint_num: int,
                input_shape: list,
                output_shape: list,
                kernel: int = 5
                ) -> [np.ndarray, np.ndarray]:
    """
    Get Results for Evaluation

    Args:
        outputs (np.ndarray): Model Output
        centers (np.ndarray): Image Center Point
        scales (np.ndarray): Image Scale
        keypoint_num (int): Num of Keypoints
        input_shape (list): Input Image Shape
        output_shape (list): Output Image Shape
        kernel (list): Gaussian Kernels. Default: 5.

    Returns:
        Prediction Keypoints List, Confidence Score

    Examples:
        >>> results = get_results(np.random.rand(2, 17, 64, 48), np.array([1.5, 1.0]), np.array([0.1, 0.2]), 17, \
        [256, 192], [64, 48])
    """
    shifts = [0.25]  # Shifts Ratios
    scales *= 200
    nr_img = outputs.shape[0]
    preds = np.zeros((nr_img, keypoint_num, 2))
    maxvals = np.zeros((nr_img, keypoint_num, 1))
    for i in range(nr_img):
        score_map = outputs[i].copy()
        score_map = score_map / 255 + 0.5
        kps = np.zeros((keypoint_num, 2))
        scores = np.zeros((keypoint_num, 1))
        border = 10
        dr = np.zeros((keypoint_num, output_shape[0] + 2 * border, output_shape[1] + 2 * border))
        dr[:, border: -border, border: -border] = outputs[i].copy()
        for w in range(keypoint_num):
            dr[w] = cv2.GaussianBlur(dr[w], (kernel, kernel), 0)
        for w in range(keypoint_num):
            for j in range(len(shifts)):
                if j == 0:
                    lb = dr[w].argmax()
                    y, x = np.unravel_index(lb, dr[w].shape)
                    dr[w, y, x] = 0
                    x -= border
                    y -= border
                lb = dr[w].argmax()
                py, px = np.unravel_index(lb, dr[w].shape)
                dr[w, py, px] = 0
                px -= border + x
                py -= border + y
                ln = (px ** 2 + py ** 2) ** 0.5
                if ln > 1e-3:
                    x += shifts[j] * px / ln
                    y += shifts[j] * py / ln
            x = max(0, min(x, output_shape[1] - 1))
            y = max(0, min(y, output_shape[0] - 1))
            kps[w] = np.array([x * 4 + 2, y * 4 + 2])
            scores[w, 0] = score_map[w, int(round(y) + 1e-9), int(round(x) + 1e-9)]

        kps[:, 0] = kps[:, 0] / input_shape[1] * scales[i][0] + centers[i][0] - scales[i][0] * 0.5
        kps[:, 1] = kps[:, 1] / input_shape[0] * scales[i][1] + centers[i][1] - scales[i][1] * 0.5
        preds[i] = kps
        maxvals[i] = scores

    return preds, maxvals


def compute_on_dataset(model,
                       dataloader,
                       flip_pairs: list,
                       keypoint_num: int,
                       input_shape: list,
                       output_shape: list,
                       ) -> list:
    """
    Inference

    Args:
        model: MSPN Model Object
        dataloader: Dataloader for data
        flip_pairs (list): Keypoints Pairs Index
        keypoint_num (int): Num of Keypoints
        input_shape (list): Input Image Shape
        output_shape (list): Output Image Shape

    Returns:
        Prediction Results

    Examples:
        >>> results = compute_on_dataset(model, dataloader, [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], \
        [13, 14], [15, 16]], 17, [256, 192], [64, 48])
    """
    results = list()
    data = tqdm(dataloader)
    for _, batch in enumerate(data):
        imgs, scores, centers, scales, img_ids = batch
        outputs = model.predict(imgs)
        outputs = outputs.asnumpy()

        imgs_flipped = np.flip(imgs.asnumpy(), 3).copy()
        imgs_flipped = Tensor.from_numpy(imgs_flipped)
        outputs_flipped = model.predict(imgs_flipped)
        outputs_flipped = outputs_flipped.asnumpy()
        outputs_flipped = flip_back(outputs_flipped, flip_pairs)
        outputs = (outputs + outputs_flipped) * 0.5

        centers = np.array(centers)
        scales = np.array(scales)
        preds, maxvals = get_results(outputs, centers, scales, keypoint_num, input_shape, output_shape)

        kp_scores = maxvals.squeeze().mean(axis=1)
        preds = np.concatenate((preds, maxvals), axis=2)

        for i in range(preds.shape[0]):
            keypoints = preds[i].reshape(-1).tolist()
            score = scores[i] * kp_scores[i]
            image_id = img_ids[i]
            results.append(dict(image_id=image_id.asnumpy().tolist(),
                                category_id=1,
                                keypoints=keypoints,
                                score=score.asnumpy().tolist()))

    return results


def evaluate(val_gt_path: str,
             pred_path: str
             ) -> None:
    """
    Evaluate Results

    Args:
        val_gt_path (str): Evaluation Ground Truth Path
        pred_path (str): Prediction Result JSON File Path

    Examples:
        >>> evaluate('./gt.json', 'res.json')
    """
    coco = COCO(val_gt_path)
    pred = coco.loadRes(pred_path)
    coco_eval = COCOeval(coco, pred, iouType='keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def draw_line(img, p1, p2):
    """
    Draw Keypoints Line

    Args:
        img (np.ndarray): Input Origin Image
        p1 (np.ndarray): Source Point
        p2 (np.ndarray): Destination Point

    Examples:
        >>> draw_line(np.random.rand(3, 64, 64), np.array([1.5, 1.6]), np.array([1.7, 1.8]))
    """
    c = (0, 0, 255)
    if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), c, 2)


def visualize_person(img, keypoint_num, joints, score=None):
    """
    Visualization for One Single Person

    Args:
        img (np.ndarray): Input Origin Image
        keypoint_num (int): Num of Keypoints
        joints (np.ndarray): Keypoints Joints
        score (float): Confidence Score. Default: None.

    Examples:
        >>> img_mod = visualize_person(np.random.rand(3, 256, 256), 17, np.array(17, 3))
    """
    pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
             [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    color = np.random.randint(0, 256, (keypoint_num, 3)).tolist()

    for i in range(keypoint_num):
        if joints[i, 0] > 0 and joints[i, 1] > 0:
            cv2.circle(img, (int(joints[i, 0]), int(joints[i, 1])), 2, tuple(color[i]), 2)
    if score:
        cv2.putText(img, str(score)[:4], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 255, 0), 2)

    for pair in pairs:
        draw_line(img, joints[pair[0] - 1], joints[pair[1] - 1])


def visualize(results, infer_dir, save_dir, keypoint_num=17, score_thre=0.8):
    """
    Visualization

    Args:
        results (list): MSPN Model Prediction Output
        infer_dir (str): Infer Image Directory
        save_dir (str): Infer Image Save Directory
        keypoint_num (int): Keypoints Number. Default: 17.
        score_thre (float): Confidence Score Threshold. Default: 0.8.

    Examples:
        >>> img_mod = visualize(results, './infer', './save')
    """
    img_ids = [str(i.split('.')[0]) for i in os.listdir(infer_dir) if i.endswith('.jpg')]
    for img_id in img_ids:
        img_name = img_id + '.jpg'
        img_path = os.path.join(infer_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        sub_results = [res for res in results if str(res['image_id']) == img_id and res['score'] > score_thre]
        for res in sub_results:
            keypoints = np.array(res['keypoints']).reshape(keypoint_num, -1)[:, :2]
            visualize_person(img, keypoint_num, keypoints)

        cv2.imwrite(os.path.join(save_dir, img_id + '_res.jpg'), img)
