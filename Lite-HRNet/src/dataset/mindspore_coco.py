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

"""COCO keypoint dataset for mindspore"""

from __future__ import absolute_import, division, print_function

import copy
import logging
import os
import random
from collections import OrderedDict, defaultdict

import cv2
import json_tricks as json
import mindspore
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.nms import oks_nms, soft_oks_nms
from utils.utils import affine_transform, fliplr_joints, get_affine_transform

logger = logging.getLogger(__name__)

class COCODataset():
    """
    COCO keypoint dataset
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }
    "skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]

    Args:
        cfg (class COCOconfig): Configures for creating dataset.
        root (str): Root dir.
        image_set (str): train2017, val2017 or test2017.
        is_train (bool): Whether on training mode.
        transform (mindvision transform): Transform performed on the original image. Default: None.
    """

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.image_thre = cfg.image_thre
        self.soft_nms = cfg.soft_nms
        self.oks_thre = cfg.oks_thre
        self.in_vis_thre = cfg.in_vis_thre
        self.bbox_file = cfg.coco_bbox_file
        self.use_gt_bbox = cfg.use_gt_bbox
        self.image_width = cfg.image_size[0]
        self.image_height = cfg.image_size[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []
        self.is_train = is_train
        self.root = root
        self.image_set = image_set
        self.output_path = cfg.output_dir
        self.data_format = cfg.data_format
        self.scale_factor = cfg.scale_factor
        self.rotation_factor = cfg.rot_fatcor
        self.flip = cfg.flip
        self.num_joints_half_body = cfg.num_joints_half_body
        self.prob_half_body = cfg.prob_half_body
        self.color_rgb = cfg.color_rgb
        self.target_type = cfg.target_type
        self.image_size = np.array(cfg.image_size)
        self.heatmap_size = np.array(cfg.heatmap_size)
        self.sigma = cfg.sigma
        self.joints_weight = 1
        self.transform = transform
        self.db = []
        self.coco = COCO(self._get_ann_file_keypoint())

        # deal with class names
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = {
            self._class_to_coco_ind[cls]: self._class_to_ind[cls]
            for cls in self.classes[1:]
            }

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)

        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

        self.db = self._get_db()

        if is_train and cfg.select_data:
            self.db = self.select_data(self.db)

    def half_body_transform(self, joints, joints_vis):
        """
        Half body transform for keypoint data augmentation.

        Args:
            joints (numpy.ndarray): Joints data.
            joints_vis (numpy.ndarray): Joints visibility.

        Returns:
            numpy.ndarray: Center of heatmap.
            scale: Scale of heatmap.
        """

        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''


        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                    and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                    )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input_data = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input_data = self.transform(input_data)
            input_data = input_data[0]

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)


        target = mindspore.Tensor(target).asnumpy()
        target_weight = mindspore.Tensor(target_weight).asnumpy()

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        if "train" in self.image_set:
            return input_data, target, target_weight

        return input_data, target, target_weight, meta

    def select_data(self, db):
        """
        Select data from database.

        Args:
            db (list): Database.

        Returns:
            list, database containing selected data.
        """

        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std ** 2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0 * (diff_norm2 ** 2) / ((0.2) ** 2 * 2.0 * area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        return db_selected

    def generate_target(self, joints, joints_vis):
        """
        Generate target gaussian heatmap.

        Args:
            joints (numpy.ndarray):  Joints data.
            joints_vis (numpy.ndarray): Joints visibility.

        Returns:
            numpy.ndarray, target heatmap.
            numpy.ndarray, target weight (1: visible, 0: invisible).
        """

        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]


        return target, target_weight

    def _get_ann_file_keypoint(self):
        """
        Get annotation file path.

        Returns:
            str, annotation file path.
        """

        prefix = 'person_keypoints' if 'test' not in self.image_set else 'image_info'
        return os.path.join(
            self.root,
            'annotations',
            prefix + '_' + self.image_set + '.json'
        )

    def _load_image_set_index(self):
        """
        Load image id using pycocotools API.

        Returns:
            int, image id index.
        """

        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        """
        Get ground truth database.

        Returns:
            list, db entry.
        """

        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """
        Ground truth bbox and keypoints.

        Returns:
            list, ground truth annotation information.
        """

        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        print(len(gt_db))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id'].
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training.
        bbox:
            [x1, y1, w, h]

        Args:
            index (int): Coco image id.

        Returns:
            list, db entry.
        """

        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        ann_ids = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec

    def _box2cs(self, box):
        """
        Change bbox parameters to center and scale.

        Args:
            box (list): bbox.

        Returns:
            numpy.ndarray, Center of bbox.
            numpy.ndarray, Scale of bbox.
        """

        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        """
        Transform XYWH bbox data to center and scale.

        Args:
            x, y, w, h (float): Bbox parameters.

        Returns:
            numpy.ndarray, center of bbox.
            numpy.ndarray, scale of bbox.
        """

        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """
        Get image path from index.

        Args:
            index (int): Index.

        Returns:
            str, image path.
        """

        file_name = '%012d.jpg' % index
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name

        prefix = 'test2017' if 'test' in self.image_set else self.image_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(
            self.root, data_name, file_name)

        return image_path

    def _load_coco_person_detection_results(self):
        """
        Load bbox from detection results.

        Returns:
            list, db entry.
        """

        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load fail!')
            return None


        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        return kpt_db

    def evaluate(self, rank, preds, output_dir, all_boxes, img_path):
        """
        Evaluate the model performance.

        Args:
            rank (int): An index for output dir to store different json results.
            preds (Tensor): Predicted heatmap output.
            output_dir (str): The place to store json results.
            all_boxes (numpy.ndarray): Bbox for all evaluating images.
            img_path (str): The list of paths of evaluating images.

        Returns:
            str, a series of indicators, e.g., AP, AP50 and AP75.
        """

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, rank)
        )

        # person x (keypoints)
        inner_kpts = []
        for idx, kpt in enumerate(preds):
            print(img_path[idx])
            inner_kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-16:-4])
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in inner_kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )

            if not keep:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[inner_keep] for inner_keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)
        if 'test' not in self.image_set:
            info_str = self._do_python_keypoint_eval(res_file)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """
        Write evaluation result to json

        Args:
            keypoints (list): Keypoints data
            res_file (str): The path to store json files

        Returns: None
        """

        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
            json.load(open(res_file))

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """
        Organize keypoints result

        Args:
            data_pack (dict): A pack of keypoints information

        Returns:
            list, organized result information
        """

        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if not img_kpts:
                continue

            inner_key_points = np.array([img_kpts[k]['keypoints']
                                         for k in range(len(img_kpts))])
            key_points = np.zeros(
                (inner_key_points.shape[0], self.num_joints * 3), dtype=np.float
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = inner_key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = inner_key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = inner_key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """
        Evaluate the result from json files, using pycocotools API
        Args:
            res_file: The path of result json files

        Returns:
            str, indicators like AP and AP50
        """

        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str
