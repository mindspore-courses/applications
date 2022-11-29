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
"""sceneflow data utils. """

import os
import re
import numpy as np



def future_optical_flow_path(root, scene, side, frame):
    """
    Generate IntoFuture optical flow path, base on the scene and frame.

    Args:
        root (str): root data path.
        scene (str): scene name
        side (str): left or right.
        frame (str): number of frame.

    Returns:
        optical flow path.

    Examples:
        >>> fofp = future_optical_flow_path('./monkaa', 'a_rain_of_stones_x2', 'left', '0001')
    """
    return os.path.join(root, "optical_flow", scene, "into_future", side,
                        f"OpticalFlowIntoFuture_{frame}_{side[0].upper()}.pfm")


def past_optical_flow_path(root, scene, side, frame):
    """
    Generate IntoPast optical flow path, base on the scene and frame.

    Args:
        root (str): root data path.
        scene (str): scene name
        side (str): left or right.
        frame (str): number of frame.

    Returns:
        optical flow path.

    Examples:
        >>> fofp = past_optical_flow_path('./monkaa', 'a_rain_of_stones_x2', 'left', '0001')
    """
    return os.path.join(root, "optical_flow", scene, "into_past", side,
                        f"OpticalFlowIntoPast_{frame}_{side[0].upper()}.pfm")


def motion_boundaries_path(root, scene, side, frame):
    """
    Generate motion boundaries path, base on the scene and frame.

    Args:
        root (str): root data path.
        scene (str): scene name
        side (str): left or right.
        frame (str): number of frame.

    Returns:
        motion boundaries path

    Examples:
        >>> fofp = motion_boundaries_path('./monkaa', 'a_rain_of_stones_x2', 'left', '0001')
    """
    return os.path.join(root, "motion_boundaries", scene, "into_past", side,
                        f"{frame}.pgm")


def read_flow(name):
    """
    Load optical flow file.

    Args:
        name (str): optical flow file path.

    Returns:
        arrays: optical flow matrix

    Examples:
        >>> optical_flow = read_flow('OpticalFlowIntoFuture_0000_L.pfm')
    """
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return read_pfm(name)[0][:, :, 0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


def read_pfm(file):
    """
    Load optical flow file in PFM suffix.

    Args:
        name (str): optical flow file path.

    Returns:
        arrays: optical flow matrix

    Examples:
        >>> optical_flow = read_pfm('OpticalFlowIntoFuture_0000_L.pfm')
    """
    file = open(file, 'rb')

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
