"""streamer"""
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
import numpy as np


def aug_matrix(w_1, h_1, w_2, h_2):
    """aug matrix"""
    d_x = (w_2 - w_1) / 2.0
    d_y = (h_2 - h_1) / 2.0

    matrix_trans = np.array([[1.0, 0, d_x], [0, 1.0, d_y], [0, 0, 1.0]])

    scale = np.min([float(w_2) / w_1, float(h_2) / h_1])

    m_p = get_affine_matrix(center=(w_2 / 2.0, h_2 / 2.0), translate=(0, 0), scale=scale)

    m_p = np.array(m_p + [0.0, 0.0, 1.0]).reshape(3, 3)
    m_p = m_p.dot(matrix_trans)

    return m_p


def get_affine_matrix(center, translate, scale):
    """get affine matrix"""
    c_x, c_y = center
    t_x, t_y = translate

    m_p = [1, 0, 0, 0, 1, 0]
    m_p = [x * scale for x in m_p]

    # Apply translation and of center translation: RSS * C^-1
    m_p[2] += m_p[0] * (-c_x) + m_p[1] * (-c_y)
    m_p[5] += m_p[3] * (-c_x) + m_p[4] * (-c_y)

    # Apply center translation: T * C * RSS * C^-1
    m_p[2] += c_x + t_x
    m_p[5] += c_y + t_y
    return m_p
