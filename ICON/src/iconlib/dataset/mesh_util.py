"""mesh related utils"""
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
import os.path as osp
import os
import numpy as np
from mindspore import Tensor, ops
import mindspore as ms
from scipy.spatial import cKDTree
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
import torchvision
from PIL import Image, ImageFont, ImageDraw
# from kaolin.metrics.trianglemesh import point_to_mesh_distance
# from kaolin.ops.mesh import check_sign
import torch

from iconlib.common.render_utils import Pytorch3dRasterizer, face_vertices


class SMPLX:
    """SMPL model"""
    def __init__(self):
        self.current_dir = osp.join(osp.dirname(__file__), "../../data/smpl_related")

        self.smpl_verts_path = osp.join(self.current_dir, "smpl_data/smpl_verts.npy")
        self.smplx_verts_path = osp.join(self.current_dir, "smpl_data/smplx_verts.npy")
        self.faces_path = osp.join(self.current_dir, "smpl_data/smplx_faces.npy")
        self.cmap_vert_path = osp.join(self.current_dir, "smpl_data/smplx_cmap.npy")

        self.faces = np.load(self.faces_path)
        self.verts = np.load(self.smplx_verts_path)
        self.smpl_verts = np.load(self.smpl_verts_path)

        self.model_dir = osp.join(self.current_dir, "models")
        self.tedra_dir = osp.join(self.current_dir, "../tedra_data")

    def get_smpl_mat(self, vert_ids):
        """get smpl model mat"""
        mat = Tensor(np.load(self.cmap_vert_path), dtype=ms.float32)
        return mat[vert_ids, :]

    def smpl2smplx(self, vert_ids=None):
        """convert vert_ids in smpl to vert_ids in smplx

        Args:
            vert_ids ([int.array]): [n, knn_num]
        """
        smplx_tree = cKDTree(self.verts, leafsize=1)
        _, ind = smplx_tree.query(self.smpl_verts, k=1)  # ind: [smpl_num, 1]

        if vert_ids is not None:
            smplx_vert_ids = ind[vert_ids]
        else:
            smplx_vert_ids = ind

        return smplx_vert_ids

    def smplx2smpl(self, vert_ids=None):
        """convert vert_ids in smplx to vert_ids in smpl

        Args:
            vert_ids ([int.array]): [n, knn_num]
        """
        smpl_tree = cKDTree(self.smpl_verts, leafsize=1)
        _, ind = smpl_tree.query(self.verts, k=1)  # ind: [smplx_num, 1]
        if vert_ids is not None:
            smpl_vert_ids = ind[vert_ids]
        else:
            smpl_vert_ids = ind

        return smpl_vert_ids


def get_visibility(xy, z, faces):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [N,2]
        z (torch.tensor): [N,1]
        faces (torch.tensor): [N,3]
        size (int): resolution of rendered image
    """

    xyz = torch.cat((xy, -z), dim=1)
    xyz = (xyz + 1.0) / 2.0
    faces = faces.long()

    rasterizer = Pytorch3dRasterizer(image_size=2 ** 12)
    meshes_screen = Meshes(verts=xyz[None, ...], faces=faces[None, ...])
    raster_settings = rasterizer.raster_settings

    pix_to_face = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(z.shape[0], 1))
    vis_mask[vis_vertices_id] = 1.0

    # print("------------------------\n")
    # print(f"keep points : {vis_mask.sum()/len(vis_mask)}")

    return vis_mask


def get_optim_grid_image(per_loop_lst, loss=None, nrow=4, type_p="smpl"):
    """get optimized grid image"""
    font_path = os.path.join(os.path.dirname(__file__), "tbfo.ttf")
    print(font_path)
    font = ImageFont.truetype(font_path, 30)
    grid_img = torchvision.utils.make_grid(torch.cat(per_loop_lst, dim=0), nrow=nrow)
    grid_img = Image.fromarray(
        ((grid_img.permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 0.5 * 255.0).astype(
            np.uint8
        )
    )

    # add text
    draw = ImageDraw.Draw(grid_img)
    grid_size = 512
    if loss is not None:
        draw.text((10, 5), f"error: {loss:.3f}", (255, 0, 0), font=font)

    if type_p == "smpl":
        for col_id, col_txt in enumerate(
                ["image", "smpl-norm(render)", "cloth-norm(pred)", "diff-norm", "diff-mask"]
        ):
            draw.text((10 + (col_id * grid_size), 5), col_txt, (255, 0, 0), font=font)
    elif type_p == "cloth":
        for col_id, col_txt in enumerate(
                ["image", "cloth-norm(recon)", "cloth-norm(pred)", "diff-norm"]
        ):
            draw.text((10 + (col_id * grid_size), 5), col_txt, (255, 0, 0), font=font)
        for col_id, col_txt in enumerate(["0", "90", "180", "270"]):
            draw.text(
                (10 + (col_id * grid_size), grid_size * 2 + 5),
                col_txt,
                (255, 0, 0),
                font=font,
            )
    else:
        print(f"{type_p} should be 'smpl' or 'cloth'")

    grid_img = grid_img.resize((grid_img.size[0], grid_img.size[1]), Image.ANTIALIAS)

    return grid_img


def orthogonal(points, calibrations, transforms=None):
    """
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    """
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def cal_sdf_batch(verts, faces, cmaps, vis, points):
    """calculate sdf batch"""
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    # cmaps [B, N_vert, 3]

    b_size = points.shape[0]

    normals = Meshes(verts, faces).verts_normals_padded()

    triangles = face_vertices(verts, faces)
    normals = face_vertices(normals, faces)
    cmaps = face_vertices(cmaps, faces)
    vis = face_vertices(vis, faces)

    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    closest_triangles = torch.gather(
        triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)
    ).view(-1, 3, 3)
    closest_normals = torch.gather(
        normals, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)
    ).view(-1, 3, 3)
    closest_cmaps = torch.gather(
        cmaps, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)
    ).view(-1, 3, 3)
    closest_vis = torch.gather(
        vis, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 1)
    ).view(-1, 3, 1)
    bary_weights = barycentric_coordinates_of_projection(
        points.view(-1, 3), closest_triangles
    )

    pts_cmap = (closest_cmaps * bary_weights[:, :, None]).sum(1).unsqueeze(0)
    pts_vis = (closest_vis * bary_weights[:, :, None]).sum(1).unsqueeze(0).ge(1e-1)
    pts_norm = (closest_normals * bary_weights[:, :, None]).sum(1).unsqueeze(
        0
    ) * torch.tensor([-1.0, 1.0, -1.0]).type_as(normals)
    pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))

    pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

    return (
        pts_sdf.view(b_size, -1, 1),
        pts_norm.view(b_size, -1, 3),
        pts_cmap.view(b_size, -1, 3),
        pts_vis.view(b_size, -1, 1),
    )


def barycentric_coordinates_of_projection(points, vertices):
    """
    https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py

    Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.

    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf

    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined
     by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    # (p, q, u, v)
    v_0, v_1, v_2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    p_p = points

    q_p = v_0
    u_p = v_1 - v_0
    v_p = v_2 - v_0
    n_p = torch.cross(u_p, v_p)
    s_p = torch.sum(n_p * n_p, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s_p[s_p == 0] = 1e-6
    one_over_4a_squared = 1.0 / s_p
    w_p = p_p - q_p
    b_2 = torch.sum(torch.cross(u_p, w_p) * n_p, dim=1) * one_over_4a_squared
    b_1 = torch.sum(torch.cross(w_p, v_p) * n_p, dim=1) * one_over_4a_squared
    weights = torch.stack((1 - b_1 - b_2, b_1, b_2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights


def feat_select(feat, select):
    """feat select"""
    # feat [B, featx2, N]
    # select [B, 1, N]
    # return [B, feat, N]

    dim = feat.shape[1] // 2
    idx = ops.tile((1 - select), (1, dim, 1)) * dim + ops.expand_dims(
        ops.expand_dims(ms.numpy.arange(0, dim), axis=0), axis=2
    ).astype(select.dtype)
    return ops.gather_elements(feat, 1, idx.astype(ms.int64))
