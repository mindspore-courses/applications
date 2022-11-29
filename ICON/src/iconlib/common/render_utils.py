"""render utils"""
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
import math
from typing import NewType
import torch
from torch import nn
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes

Tensor = NewType("Tensor", torch.Tensor)


def solid_angles(points: Tensor, triangles: Tensor) -> Tensor:
    """ Compute solid angle between the input points and triangles
        Follows the method described in:
        The Solid Angle of a Plane Triangle
        A. VAN OOSTEROM AND J. STRACKEE
        IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING,
        VOL. BME-30, NO. 2, FEBRUARY 1983
        Parameters
        -----------
            points: BxQx3
                Tensor of input query points
            triangles: BxFx3x3
                Target triangles
            thresh: float
                float threshold
        Returns
        -------
            solid_angles: BxQxF
                A tensor containing the solid angle between all query points
                and input triangles
    """
    # Center the triangles on the query points. Size should be BxQxFx3x3
    centered_tris = triangles[:, None] - points[:, :, None, None]

    # BxQxFx3
    norms = torch.norm(centered_tris, dim=-1)

    # Should be BxQxFx3
    cross_prod = torch.cross(
        centered_tris[:, :, :, 1], centered_tris[:, :, :, 2], dim=-1
    )
    # Should be BxQxF
    numerator = (centered_tris[:, :, :, 0] * cross_prod).sum(dim=-1)
    del cross_prod

    dot01 = (centered_tris[:, :, :, 0] * centered_tris[:, :, :, 1]).sum(dim=-1)
    dot12 = (centered_tris[:, :, :, 1] * centered_tris[:, :, :, 2]).sum(dim=-1)
    dot02 = (centered_tris[:, :, :, 0] * centered_tris[:, :, :, 2]).sum(dim=-1)
    del centered_tris

    denominator = (
        norms.prod(dim=-1)
        + dot01 * norms[:, :, :, 2]
        + dot02 * norms[:, :, :, 1]
        + dot12 * norms[:, :, :, 0]
    )
    del dot01, dot12, dot02, norms

    # Should be BxQ
    solid_angle = torch.atan2(numerator, denominator)
    del numerator, denominator

    torch.cuda.empty_cache()

    return 2 * solid_angle


def winding_numbers(points: Tensor, triangles: Tensor) -> Tensor:
    """ Uses winding_numbers to compute inside/outside
        Robust inside-outside segmentation using generalized winding numbers
        Alec Jacobson,
        Ladislav Kavan,
        Olga Sorkine-Hornung
        Fast Winding Numbers for Soups and Clouds SIGGRAPH 2018
        Gavin Barill
        NEIL G. Dickson
        Ryan Schmidt
        David I.W. Levin
        and Alec Jacobson
        Parameters
        -----------
            points: BxQx3
                Tensor of input query points
            triangles: BxFx3x3
                Target triangles
            thresh: float
                float threshold
        Returns
        -------
            winding_numbers: BxQ
                A tensor containing the Generalized winding numbers
    """
    # The generalized winding number is the sum of solid angles of the point
    # with respect to all triangles.
    return (
        1 / (4 * math.pi) * solid_angles(points, triangles).sum(dim=-1)
    )


def batch_contains(verts, faces, points):
    """batch contains"""
    b_p = verts.shape[0]
    n_p = points.shape[1]

    verts = verts.detach().cpu()
    faces = faces.detach().cpu()
    points = points.detach().cpu()
    contains = torch.zeros(b_p, n_p)

    for i in range(B):
        contains[i] = torch.as_tensor(
            trimesh.Trimesh(verts[i], faces[i]).contains(points[i])
        )

    return 2.0 * (contains - 0.5)


def dict2obj(d):
    """dict to obj"""
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class CreateObject():
        """create object"""

    created_object = CreateObject()
    for k in d:
        created_object.__dict__[k] = dict2obj(d[k])
    return created_object


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    batch_size, number_vertices = vertices.shape[:2]
    batch_size = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(batch_size, dtype=torch.int32).to(device)
                     * number_vertices)[:, None, None]
    vertices = vertices.reshape((batch_size * number_vertices, vertices.shape[-1]))

    return vertices[faces.long()]


class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            "image_size": image_size,
            "blur_radius": 0.0,
            "faces_per_pixel": 1,
            "bin_size": None,
            "max_faces_per_bin": None,
            "perspective_correct": True,
            "cull_backfaces": True,
        }
        raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        """forward to calculate"""
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        zbuf = zbuf + 2
        dists = dists +2
        vismask = (pix_to_face > -1).float()
        d_p = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(
            attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1]
        )
        n_p, h_p, w_p, k_p, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(n_p * h_p * w_p * k_p, 1, 1).expand(n_p * h_p * w_p * k_p, 3, d_p)
        pixel_face_vals = attributes.gather(0, idx).view(n_p * h_p * w_p * k_p, 3, d_p)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals
