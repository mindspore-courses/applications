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
Scripts to process the raw scannet dataset and generate tsdf ground truth.
"""

import os
import pickle
import argparse
import time

from tqdm import tqdm
import ray
import mindspore.dataset as ds
import numpy as np

from tools.tsdf_fusion.fusion import TSDFVolume, get_view_frustum, meshwrite, pcwrite
from tools.simple_loader import ScanNetDataset


def parse_args():
    """
    Parameter configuration
    """

    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf')
    parser.add_argument("--dataset", default='scannet')
    parser.add_argument("--data_path", metavar="DIR",
                        help="path to raw dataset", default='/data/scannet/output/')
    parser.add_argument("--save_name", metavar="DIR",
                        help="file name", default='all_tsdf')
    parser.add_argument('--test', action='store_true',
                        help='prepare the test set')
    parser.add_argument('--max_depth', default=3., type=float,
                        help='mask out large depth values since they are noisy')
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--margin', default=3, type=int)
    parser.add_argument('--voxel_size', default=0.04, type=float)

    parser.add_argument('--window_size', default=9, type=int)
    parser.add_argument('--min_angle', default=15, type=float)
    parser.add_argument('--min_distance', default=0.1, type=float)

    # ray multi processes
    parser.add_argument('--n_proc', type=int, default=16, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=2, help='#number of gpus')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=8)
    return parser.parse_args()


opts = parse_args()
opts.save_path = os.path.join(opts.data_path, opts.save_name)


def save_tsdf_full(args, scene_path, cam_intr, depth_list, cam_pose_list, color_list, save_mesh=False, save_pc=False):
    """
    Save the information of tsdf.

    Args:
        args (argparse.Namespace): Running parameters.
        scene_path (str): Path to the scene of scannet.
        cam_intr (numpy.ndarray): The camera intrinsics matrix.
        depth_list (list[dict]): The list of depth image dicts {idx: depth_img}.
        cam_pose_list (list[dict]): The list of camera pose dicts {idx: cam_pose}.
        color_list (list[dict]): The list of color image dicts {idx: color_img}.
        save_mesh (bool): Whether to save the 3D mesh. Default: False.
        save_pc (bool): Whether to save the pointcloud. Default: False.

    Examples:
        >>> save_tsdf_full(args, scene_path, cam_intr, depth_list, cam_pose_list, color_list)
    """

    vol_bnds = np.zeros((3, 2))
    n_imgs = len(depth_list.keys())
    if n_imgs > 200:
        ind = np.linspace(0, n_imgs - 1, 200).astype(np.int32)
        image_id = np.array(list(depth_list.keys()))[ind]
    else:
        image_id = depth_list.keys()
    for idx in image_id:
        depth_im = depth_list[idx]
        cam_pose = cam_pose_list[idx]

        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol_list = []
    for l in range(args.num_layers):
        tsdf_vol_list.append(TSDFVolume(vol_bnds, voxel_size=args.voxel_size * 2 ** l, margin=args.margin))

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for idx in depth_list.keys():
        if idx % 100 == 0:
            print("{}: Fusing frame {}/{}".format(scene_path, str(idx), str(n_imgs)))
        depth_im = depth_list[idx]
        cam_pose = cam_pose_list[idx]
        if not color_list:
            color_image = None
        else:
            color_image = color_list[idx]

        # Integrate observation into voxel volume (assume color aligned with depth)
        for l in range(args.num_layers):
            tsdf_vol_list[l].integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    tsdf_info = {
        'vol_origin': tsdf_vol_list[0]._vol_origin, # pylint: disable=W0212
        'voxel_size': tsdf_vol_list[0]._voxel_size, # pylint: disable=W0212
    }
    tsdf_path = os.path.join(args.save_path, scene_path)
    if not os.path.exists(tsdf_path):
        os.makedirs(tsdf_path)

    with open(os.path.join(args.save_path, scene_path, 'tsdf_info.pkl'), 'wb') as f:
        pickle.dump(tsdf_info, f)

    for l in range(args.num_layers):
        tsdf_vol, _, _ = tsdf_vol_list[l].get_volume()
        np.savez_compressed(os.path.join(args.save_path, scene_path, 'full_tsdf_layer{}'.format(str(l))), tsdf_vol)

    if save_mesh:
        for l in range(args.num_layers):
            print("Saving mesh to mesh{}.ply...".format(str(l)))
            verts, faces, norms, colors = tsdf_vol_list[l].get_mesh()

            meshwrite(os.path.join(args.save_path, scene_path, 'mesh_layer{}.ply'.format(str(l))), verts, faces, norms,
                      colors)
    if save_pc:
        # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
        print("Saving point cloud to pc.ply...")
        point_cloud = tsdf_vol_list[l].get_point_cloud()
        pcwrite(os.path.join(args.save_path, scene_path, 'pc_layer{}.ply'.format(str(l))), point_cloud)


def save_fragment_pkl(args, scene, cam_intr, depth_list, cam_pose_list):
    """
    Save the fragments in pickle.

    Args:
        args (argparse.Namespace): Running parameters.
        scene (str): Path to the scene of scannet.
        cam_intr (numpy.ndarray): The camera intrinsics matrix.
        depth_list (list[dict]): The list of depth image dicts {idx: depth_img}.
        cam_pose_list (list[dict]): The list of camera pose dicts {idx: cam_pose}.

    Examples:
        >>> save_fragment_pkl(args, scene_path, cam_intr, depth_list, cam_pose_list)
    """

    fragments = []
    print('segment: process scene {}'.format(scene))

    # gather pose
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = np.inf
    vol_bnds[:, 1] = -np.inf

    all_ids = []
    ids = []
    all_bnds = []
    count = 0
    last_pose = None
    for idx in depth_list.keys():
        depth_im = depth_list[idx]
        cam_pose = cam_pose_list[idx]

        if count == 0:
            ids.append(idx)
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.inf
            vol_bnds[:, 1] = -np.inf
            last_pose = cam_pose
            # Compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
            count += 1
        else:
            angle = np.arccos(
                ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                    [0, 0, 1])).sum())
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(idx)
                last_pose = cam_pose
                # Compute camera view frustum and extend convex hull
                view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
                count += 1
                if count == args.window_size:
                    all_ids.append(ids)
                    all_bnds.append(vol_bnds)
                    ids = []
                    count = 0

    with open(os.path.join(args.save_path, scene, 'tsdf_info.pkl'), 'rb') as f:
        tsdf_info = pickle.load(f)

    # save fragments
    for i, _ in enumerate(all_bnds):
        if not os.path.exists(os.path.join(args.save_path, scene, 'fragments', str(i))):
            os.makedirs(os.path.join(args.save_path, scene, 'fragments', str(i)))
        fragments.append({
            'scene': scene,
            'fragment_id': i,
            'image_ids': all_ids[i],
            'vol_origin': tsdf_info['vol_origin'],
            'voxel_size': tsdf_info['voxel_size'],
        })

    with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)


@ray.remote(num_cpus=opts.num_workers + 1, num_gpus=(1 / opts.n_proc))
def process_with_single_worker(args, scannet_files):
    """
    Use distributed framework ray to generate tsdf gt in parallel.

    Args:
        args (argparse.Namespace): Running parameters.
        scannet_files (list[str]): List of the scannet scenes to be processed.

    Examples:
        >>> process_with_single_worker.remote(args, scannet_files)
    """

    for scene in tqdm(scannet_files):
        if os.path.exists(os.path.join(args.save_path, scene, 'fragments.pkl')):
            continue
        print('read from disk')

        depth_all = {}
        cam_pose_all = {}
        color_all = {}

        if args.dataset == 'scannet':
            n_imgs = len(os.listdir(os.path.join(args.data_path, scene, 'color')))
            intrinsic_dir = os.path.join(args.data_path, scene, 'intrinsic', 'intrinsic_depth.txt')
            cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
            dataset_generator = ScanNetDataset(n_imgs, scene, args.data_path, args.max_depth)

        dataset = ds.GeneratorDataset(dataset_generator, ["cam_pose", "depth_im", "color_image"],
                                      num_parallel_workers=args.loader_num_workers, python_multiprocessing=False)

        for idx, (cam_pose, depth_im, _) in enumerate(dataset.create_tuple_iterator()):
            if idx % 100 == 0:
                print("{}: read frame {}/{}".format(scene, str(idx), str(n_imgs)))

            cam_pose = cam_pose.asnumpy()
            depth_im = depth_im.asnumpy()
            if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
                continue
            depth_all.update({idx: depth_im})
            cam_pose_all.update({idx: cam_pose})

        save_tsdf_full(args, scene, cam_intr, depth_all, cam_pose_all, color_all, save_mesh=False, save_pc=False)
        save_fragment_pkl(args, scene, cam_intr, depth_all, cam_pose_all)


def split_list(input_list, n):
    """
    Split the list into n parts.

    Args:
        input_list (list[list]): List to be split.
        n (int): Number of parts to split the list

    Examples:
        >>> ret = split_list(input_list, n)
    """

    ret = [[] for _ in range(n)]
    for idx, item in enumerate(input_list):
        ret[idx % n].append(item)
    return ret


def generate_pkl(args):
    """
    Combine the fragments of different scenes into one pickle.

    Args:
        args (argparse.Namespace): Running parameters.

    Examples:
        >>> generate_pkl(args)
    """

    all_scenes = sorted(os.listdir(args.save_path))
    if not args.test:
        splits = ['train', 'val']
    else:
        splits = ['test']
    for split in splits:
        fragments = []
        with open(os.path.join(args.save_path, 'splits', 'scannetv2_{}.txt'.format(split))) as f:
            split_files = f.readlines()
        for scene in all_scenes:
            if 'scene' not in scene:
                continue
            if scene + '\n' in split_files:
                with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'rb') as f:
                    frag_scene = pickle.load(f)
                fragments.extend(frag_scene)

        with open(os.path.join(args.save_path, 'fragments_{}.pkl'.format(split)), 'wb') as f:
            pickle.dump(fragments, f)


def main(args):
    """
    The main function to run the script.
    """

    all_proc = args.n_proc * args.n_gpu
    ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)

    if args.dataset == 'scannet':
        if not args.test:
            args.data_path = os.path.join(args.data_path, 'scans')
        else:
            args.data_path = os.path.join(args.data_path, 'scans_test')
        files = sorted(os.listdir(args.data_path))
    else:
        raise NameError('error!')

    files = split_list(files, all_proc)
    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(process_with_single_worker.remote(args, files[w_idx]))
    _ = ray.get(ray_worker_ids)

    if args.dataset == 'scannet':
        generate_pkl(args)


if __name__ == "__main__":
    main(opts)
