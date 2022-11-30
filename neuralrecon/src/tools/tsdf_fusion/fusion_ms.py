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
"""
Volumetric TSDF Fusion of RGB-D Images with CUDA/CPU version.
"""

import numpy as np
from numba import njit, prange
from skimage import measure


class TSDFVolume:
    """
    Volumetric TSDF Fusion of RGB-D Images for gt tsdf generation (CUDA version).

    Args:
        vol_bnds (numpy.ndarray): An ndarray of shape (3, 2). Specifies the xyz bounds (min/max) in meters.
        voxel_size (float): The volume discretization in meters.
        use_gpu (bool): Whether to use gpu. Default: True.
        margin (int): Truncation margin. Default: 5.

    Examples:
        >>> from src.tools.tsdf_fusion.fusion import TSDFVolume
        >>> TSDFVolume(vol_bnds, voxel_size)
    """

    def __init__(self, vol_bnds, voxel_size, use_gpu=True, margin=5):
        # try:
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        fusion_gpu_mode = 1
        self.cuda = cuda
        # except Exception as err:
        #     print('Warning: {}'.format(err))
        #     print('Failed to import PyCUDA. Running fusion in CPU mode.')
        #     fusion_gpu_mode = 0

        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = margin * self._voxel_size  # truncation on SDF
        self._color_const = 256 * 256

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = np.round((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(
            order='C').astype(int)
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order='C').astype(np.float32)

        # Initialize pointers to voxel volume in CPU memory
        self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self.gpu_mode = use_gpu and fusion_gpu_mode

        # Copy voxel volumes to GPU
        if self.gpu_mode:
            self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
            self.cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
            self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
            self.cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)
            self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
            self.cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

            # Cuda kernel function (C++)
            self._cuda_src_mod = SourceModule("""
        __global__ void integrate(float * tsdf_vol,
                                  float * weight_vol,
                                  float * color_vol,
                                  float * vol_dim,
                                  float * vol_origin,
                                  float * cam_intr,
                                  float * cam_pose,
                                  float * other_params,
                                  float * color_im,
                                  float * depth_im) {
          // Get voxel index
          int gpu_loop_idx = (int) other_params[0];
          int max_threads_per_block = blockDim.x;
          int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
          int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
          int vol_dim_x = (int) vol_dim[0];
          int vol_dim_y = (int) vol_dim[1];
          int vol_dim_z = (int) vol_dim[2];
          if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
              return;
          // Get voxel grid coordinates (note: be careful when casting)
          float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
          float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
          float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
          // Voxel grid coordinates to world coordinates
          float voxel_size = other_params[1];
          float pt_x = vol_origin[0]+voxel_x*voxel_size;
          float pt_y = vol_origin[1]+voxel_y*voxel_size;
          float pt_z = vol_origin[2]+voxel_z*voxel_size;
          // World coordinates to camera coordinates
          float tmp_pt_x = pt_x-cam_pose[0*4+3];
          float tmp_pt_y = pt_y-cam_pose[1*4+3];
          float tmp_pt_z = pt_z-cam_pose[2*4+3];
          float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
          float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
          float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
          // Camera coordinates to image pixels
          int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
          int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
          // Skip if outside view frustum
          int im_h = (int) other_params[2];
          int im_w = (int) other_params[3];
          if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
              return;
          // Skip invalid depth
          float depth_value = depth_im[pixel_y*im_w+pixel_x];
          if (depth_value == 0)
              return;
          // Integrate TSDF
          float trunc_margin = other_params[4];
          float depth_diff = depth_value-cam_pt_z;
          if (depth_diff < -trunc_margin)
              return;
          float dist = fmin(1.0f,depth_diff/trunc_margin);
          float w_old = weight_vol[voxel_idx];
          float obs_weight = other_params[5];
          float w_new = w_old + obs_weight;
          weight_vol[voxel_idx] = w_new;
          tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
          
          // Integrate color
          return;
          float old_color = color_vol[voxel_idx];
          float old_b = floorf(old_color/(256*256));
          float old_g = floorf((old_color-old_b*256*256)/256);
          float old_r = old_color-old_b*256*256-old_g*256;
          float new_color = color_im[pixel_y*im_w+pixel_x];
          float new_b = floorf(new_color/(256*256));
          float new_g = floorf((new_color-new_b*256*256)/256);
          float new_r = new_color-new_b*256*256-new_g*256;
          new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
          new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
          new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
          color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
        }""")

            self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

            # Determine block/grid size on GPU
            gpu_dev = cuda.Device(0)
            self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
            n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) / float(self._max_gpu_threads_per_block)))
            grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, int(np.floor(np.cbrt(n_blocks))))
            grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(np.floor(np.sqrt(n_blocks / grid_dim_x))))
            grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
            self._max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
            self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim)) / float(
                np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block)))

        else:
            # Get voxel grid coordinates
            xv, yv, zv = np.meshgrid(
                range(self._vol_dim[0]),
                range(self._vol_dim[1]),
                range(self._vol_dim[2]),
                indexing='ij'
            )
            self.vox_coords = np.concatenate([
                xv.reshape(1, -1),
                yv.reshape(1, -1),
                zv.reshape(1, -1)
            ], axis=0).astype(int).T

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        """
        Convert voxel grid coordinates to world coordinates.

        Args:
            vol_origin (numpy.ndarray): Volume origin.
            vox_coords (numpy.ndarray): Voxel grid coordinates.
            vox_size (int): The volume discretization in meters.

        Returns:
            numpy.ndarray, world coordinates.

        Examples:
            >>> cam_pts = self.vox2world(vol_origin, vox_coords, vox_size)
        """

        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        """
        Convert camera coordinates to pixel coordinates.

        Args:
            cam_pts (numpy.ndarray): Camera coordinates.
            intr (numpy.ndarray): Intrinsics of the camera.

        Returns:
            numpy.ndarray, pixel coordinates.

        Examples:
            >>> pix = self.cam2pix(cam_pts, cam_intr)
        """

        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """
        Integrate the TSDF volume.

        Args:
            tsdf_vol (numpy.ndarray): Old tsdf volume.
            dist (numpy.ndarray): The truncation value.
            w_old (numpy.ndarray): Old weights.
            obs_weight (float): The weight to assign for the current observation.

        Returns:
            numpy.ndarray, new tsdf volume.
            numpy.ndarray, new weights.

        Examples:
            >>> tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vol, dist, w_old, obs_weight)
        """

        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
        """
        Integrate an RGB-D frame into the TSDF volume.

        Args:
          color_im (numpy.ndarray): An RGB image of shape (H, W, 3).
          depth_im (numpy.ndarray): A depth image of shape (H, W).
          cam_intr (numpy.ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (numpy.ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign for the current observation.

        Examples:
            >>> self.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
        """

        im_h, im_w = depth_im.shape

        if color_im is not None:
            # Fold RGB color image into a single channel image
            color_im = color_im.astype(np.float32)
            color_im = np.floor(color_im[..., 2] * self._color_const + color_im[..., 1] * 256 + color_im[..., 0])
            color_im = color_im.reshape(-1).astype(np.float32)
        else:
            color_im = np.array(0)

        if self.gpu_mode:  # GPU mode: integrate voxel volume (calls CUDA kernel)
            for gpu_loop_idx in range(self._n_gpu_loops):
                self._cuda_integrate(self._tsdf_vol_gpu,
                                     self._weight_vol_gpu,
                                     self._color_vol_gpu,
                                     self.cuda.InOut(self._vol_dim.astype(np.float32)),
                                     self.cuda.InOut(self._vol_origin.astype(np.float32)),
                                     self.cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                                     self.cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                                     self.cuda.InOut(np.asarray([
                                         gpu_loop_idx,
                                         self._voxel_size,
                                         im_h,
                                         im_w,
                                         self._trunc_margin,
                                         obs_weight
                                     ], np.float32)),
                                     self.cuda.InOut(color_im),
                                     self.cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                                     block=(self._max_gpu_threads_per_block, 1, 1),
                                     grid=(
                                         int(self._max_gpu_grid_dim[0]),
                                         int(self._max_gpu_grid_dim[1]),
                                         int(self._max_gpu_grid_dim[2]),
                                     )
                                     )
        else:  # CPU mode: integrate voxel volume (vectorized implementation)
            # Convert voxel grid coordinates to pixel coordinates
            cam_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
            cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))
            pix_z = cam_pts[:, 2]
            pix = self.cam2pix(cam_pts, cam_intr)
            pix_x, pix_y = pix[:, 0], pix[:, 1]

            # Eliminate pixels outside view frustum
            valid_pix = np.logical_and(pix_x >= 0,
                                       np.logical_and(pix_x < im_w,
                                                      np.logical_and(pix_y >= 0,
                                                                     np.logical_and(pix_y < im_h,
                                                                                    pix_z > 0))))
            depth_val = np.zeros(pix_x.shape)
            depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

            # Integrate TSDF
            depth_diff = depth_val - pix_z
            valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
            dist = np.minimum(1, depth_diff / self._trunc_margin)
            valid_vox_x = self.vox_coords[valid_pts, 0]
            valid_vox_y = self.vox_coords[valid_pts, 1]
            valid_vox_z = self.vox_coords[valid_pts, 2]
            w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
            self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

            # Integrate color
            old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            old_b = np.floor(old_color / self._color_const)
            old_g = np.floor((old_color - old_b * self._color_const) / 256)
            old_r = old_color - old_b * self._color_const - old_g * 256
            new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_b = np.floor(new_color / self._color_const)
            new_g = np.floor((new_color - new_b * self._color_const) / 256)
            new_r = new_color - new_b * self._color_const - new_g * 256
            new_b = np.minimum(255., np.round((w_old * old_b + obs_weight * new_b) / w_new))
            new_g = np.minimum(255., np.round((w_old * old_g + obs_weight * new_g) / w_new))
            new_r = np.minimum(255., np.round((w_old * old_r + obs_weight * new_r) / w_new))
            self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b * self._color_const + new_g * 256 + new_r

    def get_volume(self):
        """
        Return the volumes.

        Returns:
            numpy.ndarray, tsdf volume.
            numpy.ndarray, color volume.
            numpy.ndarray, weight volume.

        Examples:
            >>> tsdf_vol, color_vol, weight_vol = self.get_volume()
        """

        if self.gpu_mode:
            self.cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
            self.cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
            self.cuda.memcpy_dtoh(self._weight_vol_cpu, self._weight_vol_gpu)
        return self._tsdf_vol_cpu, self._color_vol_cpu, self._weight_vol_cpu

    def get_point_cloud(self):
        """
        Extract a pointcloud from the voxel volume.

        Returns:
            numpy.ndarray, pointcloud of the voxel volume.

        Examples:
            >>> point_cloud = self.get_point_cloud()
        """

        tsdf_vol, color_vol, _ = self.get_volume()

        # Marching cubes
        verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """
        Compute a mesh from the voxel volume using marching cubes.

        Returns:
            numpy.ndarray, verts of the mesh.
            numpy.ndarray, faces of the mesh.
            numpy.ndarray, norms of the mesh.
            numpy.ndarray, colors of the mesh.

        Examples:
            >>> verts, faces, norms, colors = self.get_mesh()
        """

        tsdf_vol, color_vol, _ = self.get_volume()
        verts, faces, norms, _ = measure.marching_cubes_lewiner(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin  # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors


def rigid_transform(xyz, transform):
    """
    Applies a rigid transform to an (N, 3) pointcloud.

    Args:
        xyz (numpy.ndarray): A pointcloud of shape (N, 3).
        transform (numpy.ndarray): The transform matrix.

    Returns:
        numpy.ndarray, transform results.

    Examples:
        >>> pts_new = rigid_transform(pts, transform)
    """

    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """
    Get corners of 3D camera view frustum of depth image.

    Args:
        depth_im (numpy.ndarray): A depth image of shape (H, W).
        cam_intr (numpy.ndarray): The camera intrinsics matrix of shape (3, 3).
        cam_pose (numpy.ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).

    Returns:
        numpy.ndarray, corners of 3D camera view frustum.

    Examples:
        >>> view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
    """

    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([
        (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
    """
    Save a 3D mesh to a polygon .ply file.

    Args:
        filename (numpy.ndarray): Path to save the 3D mesh.
        verts (numpy.ndarray): The verts of the mesh.
        faces (numpy.ndarray): The faces of the mesh.
        norms (numpy.ndarray): The norms of the mesh.
        colors (numpy.ndarray): The colors of the mesh.

    Examples:
        >>> meshwrite(filename, verts, faces, norms, colors)
    """

    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
            verts[i, 0], verts[i, 1], verts[i, 2],
            norms[i, 0], norms[i, 1], norms[i, 2],
            colors[i, 0], colors[i, 1], colors[i, 2],
        ))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def pcwrite(filename, xyzrgb):
    """
    Save a pointcloud to a polygon .ply file.

    Args:
        filename (numpy.ndarray): Path to save the pointcloud.
        xyzrgb (numpy.ndarray): A pointcloud to be saved.

    Examples:
        >>> pcwrite(filename, xyzrgb)
    """

    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))


class TSDFVolumeNumpy:
    """
    Volumetric TSDF Fusion of RGB-D Images for data transformation (CPU version).

    Args:
        voxel_dim (numpy.ndarray): The volume dimension.
        origin (numpy.ndarray): The volume origin.
        voxel_size (float): The volume discretization in meters.
        margin (int): Truncation margin. Default: 3.

    Examples:
        >>> from src.tools.tsdf_fusion.fusion import TSDFVolumeNumpy
        >>> tsdf_vol = TSDFVolumeNumpy(voxel_dim, origin, voxel_size)
    """

    def __init__(self, voxel_dim, origin, voxel_size, margin=3):
        # Define voxel volume parameters
        self._voxel_size = float(voxel_size)
        self._sdf_trunc = margin * self._voxel_size
        self._const = 256 * 256

        # Adjust volume bounds
        self._vol_dim = voxel_dim.astype(np.int64)
        self._vol_origin = origin
        self._num_voxels = np.prod(self._vol_dim)

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            np.arange(0, self._vol_dim[0]),
            np.arange(0, self._vol_dim[1]),
            np.arange(0, self._vol_dim[2]),
            indexing='ij'
        )
        self._vox_coords = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1).long().astype(np.int64)

        # Convert voxel coordinates to world coordinates
        self._world_c = self._vol_origin + (self._voxel_size * self._vox_coords)
        self._world_c = np.concatenate([
            self._world_c, np.ones((len(self._world_c), 1))], axis=1)

        self.reset()

    def reset(self):
        """
        Reset the volumes.
        """

        self._tsdf_vol = np.ones(self._vol_dim)
        self._weight_vol = np.zeros(self._vol_dim)
        self._color_vol = np.zeros(self._vol_dim)

    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight):
        """
        Integrate an RGB-D frame into the TSDF volume.

        Args:
            depth_im (numpy.ndarray): A depth image of shape (H, W).
            cam_intr (numpy.ndarray): The camera intrinsics matrix of shape (3, 3).
            cam_pose (numpy.ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
            obs_weight (float): The weight to assign to the current observation.

        Examples:
            >>> self.integrate(depth_im, cam_intr, cam_pose, obs_weight)
        """

        cam_pose = cam_pose.astype(np.float32)
        cam_intr = cam_intr.astype(np.float32)
        depth_im = depth_im.astype(np.float32)
        im_h, im_w = depth_im.shape

        # Convert world coordinates to camera coordinates
        world2cam = np.linalg.inverse(cam_pose)
        cam_c = np.matmul(world2cam, self._world_c.transpose(1, 0)).transpose(1, 0).astype(np.float32)

        # Convert camera coordinates to pixel coordinates
        fx, fy = cam_intr[0, 0], cam_intr[1, 1]
        cx, cy = cam_intr[0, 2], cam_intr[1, 2]
        pix_z = cam_c[:, 2]
        pix_x = np.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).astype(np.int64)
        pix_y = np.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).astype(np.int64)

        # Eliminate pixels outside view frustum
        valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
        valid_vox_x = self._vox_coords[valid_pix, 0]
        valid_vox_y = self._vox_coords[valid_pix, 1]
        valid_vox_z = self._vox_coords[valid_pix, 2]
        depth_val = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        # Integrate tsdf
        depth_diff = depth_val - pix_z[valid_pix]
        dist = np.clip(depth_diff / self._sdf_trunc, -np.infy, 1)
        valid_pts = (depth_val > 0) & (depth_diff >= -self._sdf_trunc)
        valid_vox_x = valid_vox_x[valid_pts]
        valid_vox_y = valid_vox_y[valid_pts]
        valid_vox_z = valid_vox_z[valid_pts]
        valid_dist = dist[valid_pts]
        w_old = self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        tsdf_vals = self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        w_new = w_old + obs_weight
        self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (w_old * tsdf_vals + obs_weight * valid_dist) / w_new
        self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

    def get_volume(self):
        """
        Return the tsdf volume and weight volume.

        Returns:
            numpy.ndarray, tsdf volume.
            numpy.ndarray, weight volume.

        Examples:
            >>> self.get_volume()
        """

        return self._tsdf_vol, self._weight_vol

    @property
    def sdf_trunc(self):
        return self._sdf_trunc

    @property
    def voxel_size(self):
        return self._voxel_size
