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
"""Utils for RetinaFace evaluate."""

import datetime
import os

import numpy as np

from utils.detection import decode_bbox, decode_landm


class DetectionEngine:
    """
    According to the input parameter configuration, the prediction results are processed and validated against the data
    set.

    Args:
        config (dict): A dictionary contains some configuration for evaluation, config['val_nms_threshold'] is the
            threshold decide whether keep or remove boxes when executing nms, config['val_confidence_threshold'] is the
            threshold decide whether a prediction is valid, config['val_iou_threshold'] is the threshold decide whether
            predict box is correspond to ground truth box, config['variance'] is used to decode the prior box to
            prediction box, config['val_predict_save_folder'] is the folder path for saving predict result,
            config['val_gt_dir'] is the
            path of ground truth.
    """

    def __init__(self, config):
        self.results = {}
        self.nms_thresh = config['val_nms_threshold']
        self.conf_thresh = config['val_confidence_threshold']
        self.iou_thresh = config['val_iou_threshold']
        self.var = config['variance']
        self.save_prefix = config['val_predict_save_folder']
        self.gt_dir = config['val_gt_dir']

    def _iou(self, a, b):
        """
        Calculate intersection of union of boxes.

        Args:
            a (numpy.ndarray): A numpy array with shape [N,4], N represents number of boxes,4 represents the x,y,width
                and height of boxes.
            b (numpy.ndarray): A numpy array with shape [M,4], M represents number of boxes,4 represents the x,y,width
                and height of boxes.

        Returns:
            A numpy ndarray with shape [N,M], means each box of a calculate IoU with each box of b.
        """
        count_a = a.shape[0]
        count_b = b.shape[0]
        max_xy = np.minimum(
            np.broadcast_to(np.expand_dims(a[:, 2:4], 1), [count_a, count_b, 2]),
            np.broadcast_to(np.expand_dims(b[:, 2:4], 0), [count_a, count_b, 2]))
        min_xy = np.maximum(
            np.broadcast_to(np.expand_dims(a[:, 0:2], 1), [count_a, count_b, 2]),
            np.broadcast_to(np.expand_dims(b[:, 0:2], 0), [count_a, count_b, 2]))
        inter = np.maximum((max_xy - min_xy + 1), np.zeros_like(max_xy - min_xy))
        inter = inter[:, :, 0] * inter[:, :, 1]
        area_a = np.broadcast_to(
            np.expand_dims(
                (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), 1),
            np.shape(inter))
        area_b = np.broadcast_to(
            np.expand_dims(
                (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1), 0),
            np.shape(inter))
        union = area_a + area_b - inter
        return inter / union

    def _nms(self, boxes, threshold=0.5):
        """
        Non-Maximum Suppression.

        Args:
            boxes (numpy.ndarray): A numpy array with shape [N,5], N represents number of boxes,5 represents the x1,y1,
                x2,y2 and score of boxes.
            threshold (float): The threshold to decide whether keep or remove boxes. Default: 0.5.

        Returns:
            A numpy ndarray, which represents the index of reserved boxes.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])
            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)
            indices = np.where(ovr <= threshold)[0]
            order = order[indices + 1]
        return reserved_boxes

    def write_result(self):
        """Save result to file."""
        import json
        t = datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            if not os.path.isdir(self.save_prefix):
                os.makedirs(self.save_prefix)
            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(self.results, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def detect(self, boxes, ldm, confs, resize, scale, ldm_scale, image_path, priors):
        """
        Input the output boxes and landmarks information from RetinaFace forward pass,
        adjust their size and coordinates,finally output predict result.

        Args:
            boxes (numpy.ndarray): Shape [N,4], means predict box information get from RetinaFace network forward pass.
            ldm (numpy.ndarray): Shape [N,10], means predict landmark information get from RetinaFace network forward
            pass.
            confs (numpy.ndarray): Shape [N],
            means predict confidence information get from RetinaFace network forward pass.
            resize (float): Resize multiple of input image.
            scale (numpy.ndarray): Scale of input image x1,y1 and x2,y2.
            ldm_scale (numpy.ndarray): Similar to argument scale, scale of landmarks, usually 5 pairs of image width and
            height.
            image_path (str): Path of input image.
            priors (numpy.ndarray): Shape [N,4], priors of input image.

        Returns:
            A map, for each image, its' map like {'image_path': a string represents image path, 'bboxes': numpy
            ndarray with shape [N,5], represents x,y,w,h and conf of N boxes, 'landmarks': numpy ndarray with shape
            [N,10], represents 5 pairs of coordinates of N boxes}
        """
        if boxes.shape[0] == 0:
            event_name, img_name = image_path.split('/')
            self.results[event_name][img_name[:-4]] = {'img_path': image_path,
                                                       'bboxes': [], 'landmarks': []}
            return
        boxes = decode_bbox(np.squeeze(boxes.asnumpy(), 0), priors, self.var)
        boxes = boxes * scale / resize
        ldm = decode_landm(np.squeeze(ldm.asnumpy(), 0), priors, self.var)
        ldm = ldm * ldm_scale / resize
        scores = np.squeeze(confs.asnumpy(), 0)[:, 1]
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        ldm = ldm[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        ldm = ldm[order]
        scores = scores[order]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self._nms(dets, self.nms_thresh)
        dets = dets[keep, :]
        ldm = ldm[keep, :]
        dets[:, 2:4] = (dets[:, 2:4].astype(np.int) - dets[:, 0:2].astype(np.int)).astype(np.float)
        dets[:, 0:4] = dets[:, 0:4].astype(np.int).astype(np.float)
        ldm[:, 0:10] = ldm[:, 0:10].astype(np.int).astype(np.float)
        event_name, img_name = image_path.split('/')
        if event_name not in self.results.keys():
            self.results[event_name] = {}
        self.results[event_name][img_name[:-4]] = {'img_path': image_path,
                                                   'bboxes': dets[:, :5].astype(np.float).tolist(),
                                                   'landmarks': ldm[:, :10].astype(np.float).tolist()}

    def _get_gt_boxes(self):
        """Use WiderFace data, read the annotation information in the validation set."""
        from scipy.io import loadmat
        gt = loadmat(os.path.join(self.gt_dir, 'wider_face_val.mat'))
        hard = loadmat(os.path.join(self.gt_dir, 'wider_hard_val.mat'))
        medium = loadmat(os.path.join(self.gt_dir, 'wider_medium_val.mat'))
        easy = loadmat(os.path.join(self.gt_dir, 'wider_easy_val.mat'))
        faceboxes = gt['face_bbx_list']
        events = gt['event_list']
        files = gt['file_list']
        hard_gt_list = hard['gt_list']
        medium_gt_list = medium['gt_list']
        easy_gt_list = easy['gt_list']
        return faceboxes, events, files, hard_gt_list, medium_gt_list, easy_gt_list

    def _norm_pre_score(self):
        """Normalize each score of the predict boxes."""
        max_score = 0
        min_score = 1
        for event in self.results:
            for name in self.results[event].keys():
                bbox = np.array(self.results[event][name]['bboxes']).astype(np.float)
                if bbox.shape[0] <= 0:
                    continue
                max_score = max(max_score, np.max(bbox[:, -1]))
                min_score = min(min_score, np.min(bbox[:, -1]))
        length = max_score - min_score
        for event in self.results:
            for name in self.results[event].keys():
                bbox = np.array(self.results[event][name]['bboxes']).astype(np.float)
                if bbox.shape[0] <= 0:
                    continue
                bbox[:, -1] -= min_score
                bbox[:, -1] /= length
                self.results[event][name]['bboxes'] = bbox.tolist()

    def _image_eval(self, predict, gt, keep, iou_thresh, section_num):
        """
        Evaluate image, input predict boxes and ground truth, return evaluate result.

        Args:
            predict (numpy.ndarray): Shape [N,5], means x,y,w,h and conf of N predict boxes.
            gt (numpy.ndarray): Shape [N,4], means x,y,w,h of N ground truth boxes.
            keep (numpy.ndarray): Shape [N], value 0 means do not keep the predict box.
            iou_thresh (float): The threshold decide whether predict box is correspond to ground truth box.
            section_num (int): Section of evaluation.

        Returns:
            A numpy ndarray, whose shape is [section_num,2], means the valid image number and correctly predicted image
            number.
        """
        copy_predict = predict.copy()
        copy_groundtruth = gt.copy()
        image_p_right = np.zeros(copy_predict.shape[0])
        image_gt_right = np.zeros(copy_groundtruth.shape[0])
        proposal = np.ones(copy_predict.shape[0])
        copy_predict[:, 2:4] = copy_predict[:, 0:2] + copy_predict[:, 2:4]
        copy_groundtruth[:, 2:4] = copy_groundtruth[:, 0:2] + copy_groundtruth[:, 2:4]
        ious = self._iou(copy_predict[:, 0:4], copy_groundtruth[:, 0:4])
        for i in range(copy_predict.shape[0]):
            gt_ious = ious[i, :]
            max_iou, max_index = gt_ious.max(), gt_ious.argmax()
            if max_iou >= iou_thresh:
                if keep[max_index] == 0:
                    image_gt_right[max_index] = -1
                    proposal[i] = -1
                elif image_gt_right[max_index] == 0:
                    image_gt_right[max_index] = 1
            right_index = np.where(image_gt_right == 1)[0]
            image_p_right[i] = len(right_index)
        image_pr = np.zeros((section_num, 2), dtype=np.float)
        for section in range(section_num):
            threshold = 1 - (section + 1) / section_num
            over_score_index = np.where(predict[:, 4] >= threshold)[0]
            if over_score_index.shape[0] <= 0:
                image_pr[section, 0] = 0
                image_pr[section, 1] = 0
            else:
                index = over_score_index[-1]
                p_num = len(np.where(proposal[0:(index + 1)] == 1)[0])
                image_pr[section, 0] = p_num
                image_pr[section, 1] = image_p_right[index]
        return image_pr

    def get_eval_result(self):
        """Validate the validation set and calculate the score."""
        self._norm_pre_score()
        facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = self._get_gt_boxes()
        section_num = 1000
        sets = ['easy', 'medium', 'hard']
        set_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
        ap_key_dict = {0: "Easy   Val AP : ", 1: "Medium Val AP : ", 2: "Hard   Val AP : "}
        ap_dict = {}
        for current_set in range(len(sets)):
            gt_list = set_gts[current_set]
            count_gt = 0
            pr_curve = np.zeros((section_num, 2), dtype=np.float)
            for i, _ in enumerate(event_list):
                event = str(event_list[i][0][0])
                image_list = file_list[i][0]
                event_predict_dict = self.results[event]
                event_gt_index_list = gt_list[i][0]
                event_gt_box_list = facebox_list[i][0]
                for j, _ in enumerate(image_list):
                    event_key = str(image_list[j][0][0])
                    try:
                        event_result = event_predict_dict[event_key]
                    except KeyError:
                        continue
                    event_boxes = event_result['bboxes']
                    predict = np.array(event_boxes).astype(np.float)
                    gt_boxes = event_gt_box_list[j][0].astype('float')
                    keep_index = event_gt_index_list[j][0]
                    count_gt += len(keep_index)
                    if gt_boxes.shape[0] <= 0 or predict.shape[0] <= 0:
                        continue
                    keep = np.zeros(gt_boxes.shape[0])
                    if keep_index.shape[0] > 0:
                        keep[keep_index - 1] = 1
                    image_pr = self._image_eval(predict, gt_boxes, keep,
                                                iou_thresh=self.iou_thresh,
                                                section_num=section_num)
                    pr_curve += image_pr
            precision = pr_curve[:, 1] / pr_curve[:, 0]
            recall = pr_curve[:, 1] / count_gt
            precision = np.concatenate((np.array([0.]), precision, np.array([0.])))
            recall = np.concatenate((np.array([0.]), recall, np.array([1.])))
            for i in range(precision.shape[0] - 1, 0, -1):
                precision[i - 1] = np.maximum(precision[i - 1], precision[i])
            index = np.where(recall[1:] != recall[:-1])[0]
            ap = np.sum((recall[index + 1] - recall[index]) * precision[index + 1])
            print(ap_key_dict[current_set] + '{:.4f}'.format(ap))
        return ap_dict
