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
""" Pre process for WiderFace dataset. """

import random
import copy

import cv2
import numpy as np


def _rand(a=0., b=1.):
    """Generate a random number between a and b."""
    return np.random.rand() * (b - a) + a


def bbox_iof(bbox_a, bbox_b):
    """
    Calculate the proportion of the intersection area of bbox_a and bbox_b to bbox_a.

    Args:
        bbox_a(numpy.ndarray):A numpy.ndarray two-dimensional array with shape (N, 4),
        where N represents the number of boxes contained in bbox_a.
        bbox_b(numpy.ndarray):A numpy.ndarray two-dimensional array with shape (K, 4),
        where K represents the number of boxes contained in bbox_b.

    Returns:
        An array whose shape is (N, K).
        An element at index (n, k) contains proportion of the intersection area of bbox_a and bbox_b to bbox_a between
        n th bounding box in bbox_a and k th bounding box in bbox_b.

    Raises:
        IndexError: If bounding boxes axis 1 of bbox_a or bbox_b do not have at least length 4.

    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    top_left = np.maximum(bbox_a[:, None, 0:2], bbox_b[:, 0:2])
    bottom_right = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])
    area_intersection = np.prod(bottom_right - top_left, axis=2) * (top_left < bottom_right).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2], axis=1)
    return area_intersection / np.maximum(area_a[:, None], 1)


def _is_iof_satisfied_constraint(box, crop_box):
    """Check whether there is at least one Crop box that contains any box."""
    iof = bbox_iof(box, crop_box)
    satisfied = np.any((iof >= 1.0))
    return satisfied


def _choose_candidate(max_trial, image_w, image_h, boxes):
    """
    Randomly generate clipping boxes for input images.

    Args:
        max_trial(int): Maximum number of attempts to generate random cropping boxes.
        image_w(int): Width of the input image.
        image_h(int): Height of the input image.
        boxes(numpy.ndarray):A numpy.ndarray two-dimensional array with shape (N, 4),
        where N represents the number of boxes.

    Returns: A list of length 1 to 3.
        Each element in the list is a tuple, and each tuple contains four elements (dx,dy, nw,nh).
        Where dx is the transverse offset of the clipping box relative to the upper-left corner
        of the original image, dy is the longitudinal offset of the clipping box relative to the upper-left corner
        of the original image, nw is the width of the clipping box, and nh is the height of the clipping
        box.

    Raises:
        IndexError: If the number of bounding boxes entered is less than 1.

    """

    # Default clipping box, that is, no clipping.
    candidates = [(0, 0, image_w, image_h)]

    # Try to generate more candidates.
    for _ in range(max_trial):

        # Decide whether generate scaled box.
        if _rand() > 0.2:
            scale = _rand(0.3, 1.0)
        else:
            scale = 1.0

        # If scaled,change the width and height of clipping box.
        nh = int(scale * min(image_w, image_h))
        nw = nh

        # Randomly move clipping box in the image.
        dx = int(_rand(0, image_w - nw))
        dy = int(_rand(0, image_h - nh))

        if boxes.shape[0] > 0:
            crop_box = np.array((dx, dy, dx + nw, dy + nh))

            # Satisfy if crop box contains boxes.
            if not _is_iof_satisfied_constraint(boxes, crop_box[np.newaxis]):
                continue
            else:
                candidates.append((dx, dy, nw, nh))
        else:
            raise IndexError("!!! annotation box is less than 1")

        # Candidates enough.
        if len(candidates) >= 3:
            break

    return candidates


def _correct_bbox_by_candidates(candidates, input_width, input_height, flip, boxes, labels, landmarks,
                                allow_outside_center):
    """
    According to the input random clipping box data, generate transformed boundary box annotation.

    Args:
        candidates(list): A list of tuples.
        Each element in the list is a tuple, and each tuple contains four elements (dx,dy, nw,nh).
        Where dx is the transverse offset of the clipping box relative to the upper-left corner
        of the original image, dy is the longitudinal offset of the clipping box relative to the upper-left corner
        of the original image, nw is the width of the clipping box, and nh is the height of the clipping
        box.
        input_width(int): The width of the transformed image.
        input_height(int): The width of the transformed image.
        flip(bool): Whether to flip left and right.
        boxes(numpy.ndarray): A numpy.ndarray two-dimensional array with shape (N, 4), where N represents the number of
        boxes.
        labels(numpy.ndarray): A numpy.ndarray two-dimensional array with shape (N, ), where labels[n] represents the
        confidence of n th box.
        landmarks(numpy.ndarray): A numpy.ndarray two-dimensional array with shape (N, 10),
        where landmarks[n] represents
        5 x,y coordinate pairs of key points corresponding to boxes.
        allow_outside_center(bool): Whether the center point of the bounding box is allowed to exceed the image.


    Returns: A list of length 1 to 3.
        Each element in the list is a tuple, and each tuple contains four elements (dx,dy, nw,nh).
        Where dx is the transverse offset of the clipping box relative to the upper-left corner
        of the original image, dy is the longitudinal offset of the clipping box relative to the upper-left corner
        of the original image, nw is the width of the clipping box, and nh is the height of the clipping
        box.

    Raises:
        Exception: All candidates can not satisfied re-correct bbox.

    """
    while candidates:

        # Ignore default candidate which do not crop if random clipping box is enough.
        if len(candidates) > 1:
            candidate = candidates.pop(np.random.randint(1, len(candidates)))
        else:
            candidate = candidates.pop(np.random.randint(0, len(candidates)))

        # Left top coordinates of random clipping box (dx,dy), and its displacement relative to the upper left corner of
        # the original image (nw,nh).
        dx, dy, nw, nh = candidate

        boxes_temp = copy.deepcopy(boxes)
        landmarks_temp = copy.deepcopy(landmarks)
        labels_temp = copy.deepcopy(labels)

        landmarks_temp = landmarks_temp.reshape([-1, 5, 2])
        if nw == nh:
            scale = float(input_width) / float(nw)
        else:
            scale = float(input_width) / float(max(nh, nw))

        # Scale keypoints and bounding box coordinates to the target image size.
        boxes_temp[:, [0, 2]] = (boxes_temp[:, [0, 2]] - dx) * scale
        boxes_temp[:, [1, 3]] = (boxes_temp[:, [1, 3]] - dy) * scale
        landmarks_temp[:, :, 0] = (landmarks_temp[:, :, 0] - dx) * scale
        landmarks_temp[:, :, 1] = (landmarks_temp[:, :, 1] - dy) * scale

        if flip:
            boxes_temp[:, [0, 2]] = input_width - boxes_temp[:, [2, 0]]
            landmarks_temp[:, :, 0] = input_width - landmarks_temp[:, :, 0]
            landms_t_1 = landmarks_temp[:, 1, :].copy()
            landmarks_temp[:, 1, :] = landmarks_temp[:, 0, :]
            landmarks_temp[:, 0, :] = landms_t_1
            landms_t_4 = landmarks_temp[:, 4, :].copy()
            landmarks_temp[:, 4, :] = landmarks_temp[:, 3, :]
            landmarks_temp[:, 3, :] = landms_t_4

        if allow_outside_center:
            pass
        else:
            mask1 = np.logical_and((boxes_temp[:, 0] + boxes_temp[:, 2]) / 2. >= 0.,
                                   (boxes_temp[:, 1] + boxes_temp[:, 3]) / 2. >= 0.)
            boxes_temp = boxes_temp[mask1]
            landmarks_temp = landmarks_temp[mask1]
            labels_temp = labels_temp[mask1]

            mask2 = np.logical_and((boxes_temp[:, 0] + boxes_temp[:, 2]) / 2. <= input_width,
                                   (boxes_temp[:, 1] + boxes_temp[:, 3]) / 2. <= input_height)
            boxes_temp = boxes_temp[mask2]
            landmarks_temp = landmarks_temp[mask2]
            labels_temp = labels_temp[mask2]

        # recorrect x, y for case x,y < 0 reset to zero, after dx and dy, some box can smaller than zero
        boxes_temp[:, 0:2][boxes_temp[:, 0:2] < 0] = 0

        # recorrect w,h not higher than input size
        boxes_temp[:, 2][boxes_temp[:, 2] > input_width] = input_width
        boxes_temp[:, 3][boxes_temp[:, 3] > input_height] = input_height
        box_width = boxes_temp[:, 2] - boxes_temp[:, 0]
        box_height = boxes_temp[:, 3] - boxes_temp[:, 1]

        # discard invalid box: w or h smaller than 1 pixel
        mask3 = np.logical_and(box_width > 1, box_height > 1)
        boxes_temp = boxes_temp[mask3]
        landmarks_temp = landmarks_temp[mask3]
        labels_temp = labels_temp[mask3]

        # normal
        boxes_temp[:, [0, 2]] /= input_width
        boxes_temp[:, [1, 3]] /= input_height
        landmarks_temp[:, :, 0] /= input_width
        landmarks_temp[:, :, 1] /= input_height

        landmarks_temp = landmarks_temp.reshape([-1, 10])
        labels_temp = np.expand_dims(labels_temp, 1)

        targets_t = np.hstack((boxes_temp, landmarks_temp, labels_temp))

        if boxes_temp.shape[0] > 0:
            return targets_t, candidate

    raise Exception('all candidates can not satisfied re-correct bbox')


def get_interp_method(interp, sizes=()):
    """
    Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and a auto-estimation method.

    Parameters
    interp : int
        interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Nearest Neighbors. [Originally it should be Area-based,
        as we cannot find Area-based, so we use NN instead.
        Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method mentioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.

    sizes : tuple of int
        (old_height, old_width, new_height, new_width), if None provided, auto(9)
        will return Area(2) anyway.

    Returns
    int
        interp method from 0 to 4
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            if nh < oh and nw < ow:
                return 0
            return 1
        return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp


def cv_image_reshape(interp):
    """Reshape pil image."""
    reshape_type = {
        0: cv2.INTER_LINEAR,
        1: cv2.INTER_CUBIC,
        2: cv2.INTER_AREA,
        3: cv2.INTER_NEAREST,
        4: cv2.INTER_LANCZOS4,
    }
    return reshape_type[interp]


class PreProcessor:
    """
    Do data preprocessing and data enhancement.According to the input original image, boundary box and key point
    annotation information, randomly clip, flip and do other data enhancement operation to the image,
    then the image size and color are normalized, shape is adjusted, finally output the adjusted
    picture and annotation data.

    Args:
        image_dim (int): After data preprocessing, the height and width of output images. For MobileNet025, it is
        typically 640; For ResNet50, it is generally 840.

    Inputs:
        image: An image to be preprocessed, the shape is (H, W, C), representing the height, width and channel of
        the image, usually can input the image read by cv2.imread function.
        annotation: A is the annotation of the input image, and the shape is (N,4+10+1).Annotation [n] represents the
        NTH boundary frame annotation
        information of the
        image, in which the first four numbers are the XY coordinates, width and height of the boundary frame,
        the last ten numbers represent the XY coordinate pairs of the key points of the five faces in the
        boundary frame,
        and the last number represents the confidence of the key frame.

    Outputs:
        A tuple whose first element is the adjusted image data and the second element is the adjusted annotation data.

    """

    def __init__(self, image_dim):
        self.image_input_size = image_dim

    def __call__(self, image, annotation):
        """Do data augment."""
        assert annotation.shape[0] > 0, "target without ground truth."
        annotation_copy = copy.deepcopy(annotation)
        boxes = annotation_copy[:, :4]
        landmarks = annotation_copy[:, 4:-1]
        labels = annotation_copy[:, -1]

        aug_image, aug_target = self.do_data_augment(image, boxes, labels, landmarks, self.image_input_size)

        return aug_image, aug_target

    def do_data_augment(self, image, boxes, labels, landmarks, image_size, max_trial=250):
        """
        The main process of data enhancement operations.

        Args:
            image: An image to be preprocessed, the shape is (H, W, C), representing the height, width and channel of
            the image, usually can input the image read by cv2.imread function.
            boxes(numpy.ndarray): A numpy.ndarray two-dimensional
            array with shape (N, 4), where N represents the number of boxes.
            labels(numpy.ndarray): A numpy.ndarray
            two-dimensional array with shape (N, ), where labels[n] represents the confidence of n th box.
            landmarks(numpy.ndarray): A numpy.ndarray two-dimensional array with shape (N, 10), where landmarks[n]
            represents 5 x,y coordinate pairs of key points corresponding to boxes.
            image_size(int):  After data preprocessing, the height and width of output images. For MobileNet025, it
            is typically 640; For ResNet50, it is generally 840.
            max_trial(int): Maximum number of attempts to generate random cropping boxes.

        Returns:
            A tuple whose first element is the adjusted image data and the second element is the adjusted
            annotation data.

        """
        image_h, image_w, _ = image.shape
        input_h, input_w = image_size, image_size

        flip = _rand() < .5

        candidates = _choose_candidate(max_trial=max_trial,
                                       image_w=image_w,
                                       image_h=image_h,
                                       boxes=boxes)
        targets, candidate = _correct_bbox_by_candidates(candidates=candidates,
                                                         input_width=input_w,
                                                         input_height=input_h,
                                                         flip=flip,
                                                         boxes=boxes,
                                                         labels=labels,
                                                         landmarks=landmarks,
                                                         allow_outside_center=False)

        # crop image
        dx, dy, nw, nh = candidate
        image = image[dy:(dy + nh), dx:(dx + nw)]

        if nw != nh:
            assert nw == image_w and nh == image_h

            # pad ori image to square
            l = max(nw, nh)
            t_image = np.empty((l, l, 3), dtype=image.dtype)
            t_image[:, :] = (104, 117, 123)
            t_image[:nh, :nw] = image
            image = t_image

        interp = get_interp_method(interp=10)
        image = cv2.resize(image, (input_w, input_h), interpolation=cv_image_reshape(interp))

        if flip:
            image = image[:, ::-1]

        image = image.astype(np.float32)
        image -= (104, 117, 123)
        image = image.transpose(2, 0, 1)

        return image, targets
