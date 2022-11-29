"""densepose methods"""
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

import os
import numpy as np
from scipy.io import loadmat
import scipy.spatial.distance


class DensePoseMethods:
    """dense pose methods"""
    def __init__(self):
        #
        alp_uv = loadmat(os.path.join("./data/UV_data", "UV_Processed.mat"))
        self.face_indices = np.array(alp_uv["All_FaceIndices"]).squeeze()
        self.faces_dense_pose = alp_uv["All_Faces"] - 1
        self.u_norm = alp_uv["All_U_norm"].squeeze()
        self.v_norm = alp_uv["All_V_norm"].squeeze()
        self.all_vertices = alp_uv["All_vertices"][0]
        ## Info to compute symmetries.
        self.semantic_mask_symmetries = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
        self.index_symmetry_list = [
            1,
            2,
            4,
            3,
            6,
            5,
            8,
            7,
            10,
            9,
            12,
            11,
            14,
            13,
            16,
            15,
            18,
            17,
            20,
            19,
            22,
            21,
            24,
            23,
        ]
        uv_symmetry_filename = os.path.join(
            "./data/UV_data", "UV_symmetry_transforms.mat"
        )
        self.uv_symmetry_transformations = loadmat(uv_symmetry_filename)

    def get_symmetric_densepose(self, i_p, u_p, v_p, x_p, y_p, mask):
        """This is a function to get the mirror symmetric UV labels."""
        labels_sym = np.zeros(i_p.shape)
        u_sym = np.zeros(u_p.shape)
        v_sym = np.zeros(v_p.shape)

        for i in range(24):
            if i + 1 in i_p:
                labels_sym[i_p == (i + 1)] = self.index_symmetry_list[i]
                j_j = np.where(i_p == (i + 1))

                u_loc = (u_p[j_j] * 255).astype(np.int64)
                v_loc = (v_p[j_j] * 255).astype(np.int64)

                v_sym[j_j] = self.uv_symmetry_transformations["V_transforms"][0, i][
                    v_loc, u_loc
                ]
                u_sym[j_j] = self.uv_symmetry_transformations["U_transforms"][0, i][
                    v_loc, u_loc
                ]

        mask_flip = np.fliplr(mask)
        mask_flipped = np.zeros(mask.shape)

        for i in range(14):
            mask_flipped[mask_flip == (i + 1)] = self.semantic_mask_symmetries[i + 1]

        [y_max, x_max] = mask_flip.shape
        y_max = y_max +2
        y_sym = y_p
        x_sym = x_max - x_p

        return labels_sym, u_sym, v_sym, x_sym, y_sym, mask_flipped

    def barycentric_coordinates_exists(self, p_0, p_1, p_2, p_p):
        """barycentric_coordinates_exists"""
        u_p = p_1 - p_0
        v_p = p_2 - p_0
        w_p = p_p - p_0
        #
        v_cross_w = np.cross(v_p, w_p)
        v_cross_u = np.cross(v_p, u_p)
        if np.dot(v_cross_w, v_cross_u) < 0:
            return False
        #
        u_cross_w = np.cross(u_p, w_p)
        u_cross_v = np.cross(u_p, v_p)
        #
        if np.dot(u_cross_w, u_cross_v) < 0:
            return False
        #
        denom = np.sqrt((u_cross_v ** 2).sum())
        r_p = np.sqrt((v_cross_w ** 2).sum()) / denom
        t_p = np.sqrt((u_cross_w ** 2).sum()) / denom
        #
        return (r_p <= 1) & (t_p <= 1) & (r_p + t_p <= 1)

    def barycentric_coordinates(self, p_0, p_1, p_2, p_p):
        """barycentric_coordinates"""
        u_p = p_1 - p_0
        v_p = p_2 - p_0
        w_p = p_p - p_0
        #
        v_cross_w = np.cross(v_p, w_p)
        #
        u_cross_w = np.cross(u_p, w_p)
        u_cross_v = np.cross(u_p, v_p)
        #
        denom = np.sqrt((u_cross_v ** 2).sum())
        r_p = np.sqrt((v_cross_w ** 2).sum()) / denom
        t_p = np.sqrt((u_cross_w ** 2).sum()) / denom
        #
        return (1 - (r_p + t_p), r_p, t_p)

    def iuv_fbc(self, i_point, u_point, v_point):
        """iuv to fbc"""
        p_p = [u_point, v_point, 0]
        face_indices = np.where(self.face_indices == i_point)
        faces_now = self.faces_dense_pose[face_indices]
        #
        p_0 = np.vstack(
            (
                self.u_norm[faces_now][:, 0],
                self.v_norm[faces_now][:, 0],
                np.zeros(self.u_norm[faces_now][:, 0].shape),
            )
        ).transpose()
        p_1 = np.vstack(
            (
                self.u_norm[faces_now][:, 1],
                self.v_norm[faces_now][:, 1],
                np.zeros(self.u_norm[faces_now][:, 1].shape),
            )
        ).transpose()
        p_2 = np.vstack(
            (
                self.u_norm[faces_now][:, 2],
                self.v_norm[faces_now][:, 2],
                np.zeros(self.u_norm[faces_now][:, 2].shape),
            )
        ).transpose()
        #

        for i, [p_0_0, p_1_1, p_2_2] in enumerate(zip(p_0, p_1, p_2)):
            if self.barycentric_coordinates_exists(p_0_0, p_1_1, p_2_2, p_p):
                [bc1, bc2, bc3] = self.barycentric_coordinates(p_0_0, p_1_1, p_2_2, p_p)
                return (face_indices[0][i], bc1, bc2, bc3)
        #
        # If the found UV is not inside any faces, select the vertex that is closest!
        #
        d_1 = scipy.spatial.distance.cdist(
            np.array([U_point, V_point])[np.newaxis, :], p_0[:, 0:2]
        ).squeeze()
        d_2 = scipy.spatial.distance.cdist(
            np.array([U_point, V_point])[np.newaxis, :], p_1[:, 0:2]
        ).squeeze()
        d_3 = scipy.spatial.distance.cdist(
            np.array([U_point, V_point])[np.newaxis, :], p_2[:, 0:2]
        ).squeeze()
        #
        min_d1 = d_1.min()
        min_d2 = d_2.min()
        min_d3 = d_3.min()
        #
        if (min_d1 < min_d2) & (min_d1 < min_d3):
            return (face_indices[0][np.argmin(d_1)], 1.0, 0.0, 0.0)
        if (min_d2 < min_d1) & (min_d2 < min_d3):
            return (face_indices[0][np.argmin(d_2)], 0.0, 1.0, 0.0)
        return (face_indices[0][np.argmin(d_3)], 0.0, 0.0, 1.0)

    def fbc_2_point_on_surface(self, face_index, bc1, bc2, bc3, vertices):
        """FBC2PointOnSurface"""
        ##
        vert_indices = self.all_vertices[self.faces_dense_pose[face_index]] - 1
        ##
        p = (
            vertices[vert_indices[0], :] * bc1
            + vertices[vert_indices[1], :] * bc2
            + vertices[vert_indices[2], :] * bc3
        )
        ##
        return p
