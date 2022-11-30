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
Dividing the original data into a training set and a test set,
and provide method to perform data augmentation on the training data set.
"""

import os
import numpy as np
import cv2


def calculate_pitch_yaw_roll(landmarks_2d: np.ndarray,
                             cam_w: int = 256,
                             cam_h: int = 256):
    """
    Return the pitch yaw and roll angles associated with the input image.

    Args:
        landmarks_2d (np.ndarray): Landmark coordinate.
        cam_w (int): Weight of image. Default: 256.
        cam_h (int): Height of image. Default: 256.

    Returns:
        map. Euler_angles contain (pitch, yaw, roll).
    """

    # Estimated camera matrix values.
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    landmarks_3d = np.float32([
        [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT,
        [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT,
        [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
        [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
        [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
        [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
        [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
        [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
        [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
        [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
        [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
        [0.000000, -7.415691, 4.070434],  # CHIN
    ])
    landmarks_2d = np.asarray(landmarks_2d, dtype=np.float32).reshape(-1, 2)

    # Applying the PnP solver to find the 3D pose of the head from the 2D position of the landmarks.
    # r_vec - Output rotation vector that, together with tvec, brings points from the world coordinate
    # system to the camera coordinate system.
    # t_vec - Output translation vector. It is the position of the world origin
    # (SELLION) in camera co-ords
    _, r_vec, t_vec = cv2.solvePnP(
        landmarks_3d, landmarks_2d, camera_matrix, camera_distortion)
    r_mat, _ = cv2.Rodrigues(r_vec)
    pose_mat = cv2.hconcat((r_mat, t_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    return map(lambda k: k[0], euler_angles)

def rotate(angle, center, landmark):
    """
    Rotation of the original landmarks in response to the angle of rotation.

    Args:
        angle (int): Landmark rotation angle.
        center (tuple): Regional centres.
        landmark (np.ndarray): In the WLFW dataset are the coordinates of 98 points.

    Returns:
        matrix (np.ndarray): Transformation matrix.
        landmark_ (np.ndarray): Landmarks after the rotation.
    """

    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)

    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0] = alpha
    matrix[0, 1] = beta
    matrix[0, 2] = (1 - alpha) * center[0] - beta * center[1]
    matrix[1, 0] = -beta
    matrix[1, 1] = alpha
    matrix[1, 2] = beta * center[0] + (1 - alpha) * center[1]

    landmark_ = np.asarray([(matrix[0, 0] *
                             x +
                             matrix[0, 1] *
                             y +
                             matrix[0, 2], matrix[1, 0] *
                             x +
                             matrix[1, 1] *
                             y +
                             matrix[1, 2]) for (x, y) in landmark])

    return matrix, landmark_


class ImageData():
    """
    Data enhancement of the training set and preservation of the processed data.

    Args:
        line (str):  A line of data in the comment file.
        img_dir (str): Catalogue of image files.
        target_dataset (str): Dataset type.
        img_size (int): Size of images.

    Outputs:
        list. A list of images and their annotated information.
    """

    def __init__(self,
                 line: str,
                 img_dir: str,
                 target_dataset: str = '300W',
                 img_size: int = 112):
        self.img_size = img_size
        self.target_dataset = target_dataset
        line = line.strip().split()
        # 0-195: landmark 196-199: bbox
        # 200: pose         0->normal pose            1->large pose
        # 201: expression   0->normal expression      1->exaggerate expression
        # 202: illumination 0->normal illumination    1->extreme illumination
        # 203: make-up      0->no make-up             1->make-up
        # 204: occlusion    0->no occlusion           1->occlusion
        # 205: blur         0->clear                  1->blur
        # 206: image name

        self.list = line
        if self.target_dataset == '300W':
            self.landmark = np.asarray(
                list(map(float, line[:136])), dtype=np.float32).reshape(-1, 2)
            self.box = np.asarray(
                list(map(int, line[136:140])), dtype=np.int32)
            flag = list(map(int, line[140:146]))
            flag = list(map(bool, flag))
            self.path = os.path.join(img_dir, line[146])
        else:
            self.landmark = np.asarray(
                list(map(float, line[:196])), dtype=np.float32).reshape(-1, 2)
            self.box = np.asarray(
                list(map(int, line[196:200])), dtype=np.int32)
            flag = list(map(int, line[200:206]))
            flag = list(map(bool, flag))
            self.path = os.path.join(img_dir, line[206])

        self.pose = flag[0]
        self.expression = flag[1]
        self.illumination = flag[2]
        self.make_up = flag[3]
        self.occlusion = flag[4]
        self.blur = flag[5]
        self.img = None
        self.imgs = []
        self.landmarks = []
        self.boxes = []

    def load_data(self,
                  train_flag: bool,
                  repeat: int,
                  mirror: str = None):
        """
        Load data from the original dataset and do data augmentation on the training data.

        Args:
            train_flag: bool.
            repeat: int. Number of data enhancements done per image.
            mirror: str. Numbered documents for landmarks.
        """

        if mirror is not None:
            with open(mirror, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == 1
                mirror_idx = lines[0].strip().split(',')
                mirror_idx = list(map(int, mirror_idx))
        xy = np.min(self.landmark, axis=0).astype(np.int32)
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh / 2).astype(np.int32)
        img = cv2.imread(self.path)
        boxsize = int(np.max(wh) * 1.2)

        # box left bottom corner
        xy = center - boxsize // 2
        x1, y1 = xy

        # box right bottom corner
        x2, y2 = xy + boxsize
        height, width, _ = img.shape

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        img_t = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0
                or edx > 0 or edy > 0):
            img_t = cv2.copyMakeBorder(
                img_t, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        img_t = cv2.resize(img_t, (self.img_size, self.img_size))
        landmark = (self.landmark - xy) / boxsize

        self.imgs.append(img_t)
        self.landmarks.append(landmark)

        if train_flag:
            # Data augmentation 10 times
            while len(self.imgs) < repeat:
                angle = np.random.randint(-30, 30)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                matrix, landmark = rotate(angle, (cx, cy), self.landmark)

                img_t = cv2.warpAffine(
                    img, matrix, (int(img.shape[1] * 1.1), int(img.shape[0] * 1.1)))

                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                size = np.random.randint(
                    int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                xy = np.asarray(
                    (cx - size // 2, cy - size // 2), dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = img_t.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                img_t = img_t[y1:y2, x1:x2]
                if (dx > 0 or dy > 0
                        or edx > 0 or edy > 0):
                    img_t = cv2.copyMakeBorder(
                        img_t, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                img_t = cv2.resize(img_t, (self.img_size, self.img_size))
                if mirror is not None and np.random.choice((True, False)):
                    landmark[:, 0] = 1 - landmark[:, 0]
                    landmark = landmark[mirror_idx]
                    img_t = cv2.flip(img_t, 1)
                self.imgs.append(img_t)
                self.landmarks.append(landmark)

    def get_data(self,
                 path: str,
                 prefix: str):
        """
        Processing of the enhanced data into (data enhanced file path, landmark,
        attribute(pose, expression, illumination,make_up,occlusion,blur),
        angle) format.

        Args:
            path (str): Image data storage path.
            prefix (str): Enhanced image names.

        Returns:
            list. A list of images and their annotated information.
        """

        attributes = [self.pose,
                      self.expression,
                      self.illumination,
                      self.make_up,
                      self.occlusion,
                      self.blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []

        if self.target_dataset == '300W':
            tracked_points = [17, 21, 22, 26, 36,
                              39, 42, 45, 31, 35, 48, 54, 57, 8]
        else:
            tracked_points = [33, 38, 50, 46, 60,
                              64, 68, 72, 55, 59, 76, 82, 85, 16]

        for i, (img, landmark) in enumerate(zip(self.imgs, self.landmarks)):
            save_path = os.path.join(path, prefix + '_' + str(i) + '.png')
            cv2.imwrite(save_path, img)

            euler_angles_landmark = []
            for index in tracked_points:
                euler_angles_landmark.append(landmark[index])
            euler_angles_landmark = np.asarray(
                euler_angles_landmark).reshape((-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(
                euler_angles_landmark[0])
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            landmark_str = ' '.join(
                list(map(str, landmark.reshape(-1).tolist())))

            label = '{} {} {} {}\n'.format(
                save_path, landmark_str, attributes_str, euler_angles_str)

            labels.append(label)
        return labels


def get_dataset_list(
        img_dir: str,
        output_dir: str,
        ld_dir: str,
        train_flag: bool,
        target_dataset: str = '300W',
        mirror_file: str = './datasets/300W/300W_annotations/Mirror68.txt'):
    """
    Generate a data set and save it to the specified directory.

    Args:
        img_dir: str. Catalogue of image data.
        output_dir: str. Storage directory for image data.
        ld_dir: str. Save path for landmarks.
        train_flag: bool. Generate markers for training or test data.

    Outputs:
        Label file, including the label information.
    """

    with open(ld_dir, 'r') as f:
        lines = f.readlines()
        labels = []
        save_img = os.path.join(output_dir, 'imgs')
        if not os.path.exists(save_img):
            os.mkdir(save_img)

        for i, line in enumerate(lines):
            img = ImageData(line, img_dir, target_dataset)
            img_name = img.path
            img.load_data(train_flag, 10, mirror_file)
            _, filename = os.path.split(img_name)
            filename, _ = os.path.splitext(filename)
            label_txt = img.get_data(save_img, str(i) + '_' + filename)
            labels.append(label_txt)
            if ((i + 1) % 100) == 0:
                print('file: {}/{}'.format(i + 1, len(lines)))

    with open(os.path.join(output_dir, 'list.txt'), 'w') as f:
        for label in labels:
            f.writelines(label)
