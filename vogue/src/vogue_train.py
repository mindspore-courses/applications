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
"""Training vogue"""

import argparse
import copy
import os
import time

import numpy as np
import PIL.Image
from mindspore import load_checkpoint, load_param_into_net
import mindspore as ms
from mindspore import Tensor, ops, nn, set_seed

from models.vogue_generator import Generator
from models.vogue_discriminator import Discriminator
from process_datasets.inshop import Inshop
from losses.vogue_loss import CustomWithLossCell, StyleGANLoss

os.environ['GLOG_v'] = '3'


def setup_snapshot_image_grid(dataset, seed=0):
    """
    Return the image grid size and images, labels, poses。

    Args:
        dataset (class): The dataset.
        seed (int): Random seed. Default: 0.

    Returns:
        tuple, the image grid size.
        numpy.ndarray, images.
        numpy.ndarray, labels.
        numpy.ndarray, poses.

    Examples:
        >>> size, images, labels, pose = setup_snapshot_image_grid(dataset)
    """
    rnd = np.random.RandomState(seed)
    gw = 4
    gh = 4

    if not dataset.has_labels:
        all_indices = list(range(len(dataset)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        label_groups = {}
        for idx in range(len(dataset)):
            label = tuple(dataset.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    data_images, data_labels, data_pose = zip(*[dataset[i] for i in grid_indices])
    grid = (gw, gh)
    out_images = np.stack(data_images)
    out_labels = np.stack(data_labels)
    out_pose = np.stack(data_pose)
    return grid, out_images, out_labels, out_pose


def save_image_grid(s_image, f_name, d_range, size):
    """
    Save grid image.

    Args:
        s_image (numpy.ndarray): The image.
        f_name (str): The file path.
        d_range (list): Image range.
        size (tuple): Image size.

    Examples:
        >>> save_image_grid(images, path, d_range=[0, 255], size=grid_size)
    """
    lo, hi = d_range
    s_image = np.asarray(s_image, dtype=np.float32)
    s_image = (s_image - lo) * (255 / (hi - lo))
    s_image = np.rint(s_image).clip(0, 255).astype(np.uint8)

    gw, gh = size
    _, cc, hh, ww = s_image.shape
    s_image = s_image.reshape(gh, gw, cc, hh, ww)
    s_image = s_image.transpose(0, 3, 1, 4, 2)
    s_image = s_image.reshape(gh * hh, gw * ww, cc)

    if cc == 1:
        PIL.Image.fromarray(s_image[:, :, 0], 'L').save(f_name)
    if cc == 3:
        PIL.Image.fromarray(s_image, 'RGB').save(f_name)


def save_model(module_list, out_dir, cur_nimg):
    """
    Save model.

    Args:
        module_list (list): A list of module names and modules.
        out_dir (str): The output directory.
        cur_nimg (int): Current image number.

    Examples:
        >>> save_image_grid(module_list, 'out/', 10)
    """
    for name, module in module_list:
        module_copy = copy.deepcopy(module)
        for key, value in module_copy.parameters_dict().items():
            if 'all_conv' in key:
                value.requires_grad = False
        module_copy.requires_grad = False
        all_param = []
        for par in module_copy.trainable_params():
            layer = dict()
            layer['name'] = par.name
            layer['data'] = par
            all_param.append(layer)
        ms.save_checkpoint(all_param, out_dir + f'network-snapshot-{cur_nimg // 1000:06d}-' + name + '.ckpt')
        del module_copy


def save_image_snapshot(out_dir, cur_nimg, model, grid_z, grid_c, grid_pose, grid_size, concat):
    """
    Save image snapshot.

    Args:
        out_dir (str): The output directory.
        cur_nimg (int): Current image number.
        model (nn.Cell): Generator_ema infer model.
        grid_z (tuple): Noise.
        gird_c (tuple): Labels.
        grid_pose (tuple): Poses.
        grid_size (tuple): The image grid size.
        concat (Operation): Concat operator.

    Examples:
        >>> save_image_snapshot(out_dir, cur_nimg, model, grid_z, grid_c, grid_pose, grid_size, concat)
    """
    param_dict = load_checkpoint(out_dir + f'network-snapshot-{cur_nimg // 1000:06d}-G_ema.ckpt')
    load_param_into_net(model, param_dict)
    images = []
    for z, c, pose in zip(grid_z, grid_c, grid_pose):
        ws = model.mapping.construct(z=z, c=c)
        image = model.synthesis.construct(ws=ws, pose=pose, noise_mode=0)
        images.append(image)
    images = concat(images).asnumpy()
    save_image_grid(images, os.path.join(out_dir, f'fakes{cur_nimg // 1000:06d}.png'), d_range=[-1, 1], size=grid_size)


def train(args):
    """
    Train the model.

    Args:
        args (argparse.Namespace): The args of train.

    Examples:
        >>> train(args)
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_context(device_id=args.device)
    out_dir = args.outdir
    snap = args.snap
    random_seed = args.seed
    data_dir = args.data_dir
    posefile = args.posefile
    mirror = args.mirror
    total_kimg = args.total_kimg
    batch_size = args.batch_size
    model_path = args.model_path

    np.random.seed(random_seed)
    set_seed(random_seed)

    image_snapshot_ticks = snap
    network_snapshot_ticks = snap
    cur_tick = 0
    batch_idx = 0
    kimg_per_tick = 4
    tick_start_nimg = 2404000
    cur_nimg = 2404000
    abort_fn = None
    all_done = False

    g_reg_interval = 4
    d_reg_interval = 16

    g_mapping_kwargs = {'num_layers': 2}
    g_synthesis_kwargs = {'channel_base': 16384, 'channel_max': 512,
                          'num_fp16_res': 4, 'conv_clamp': 256}
    d_epilogue_kwargs = {'mbstd_group_size': 4}
    g_opt_kwargs = {'class_name': 'mindspore.nn.Adam', 'lr': 0.0025, 'betas': [1e-08, 0.99], 'eps': 1e-08}
    d_opt_kwargs = {'class_name': 'mindspore.nn.Adam', 'lr': 0.0025, 'betas': [1e-08, 0.99], 'eps': 1e-08}

    concat = ops.Concat()
    split_batch = ops.Split(0, output_num=2)
    split_gpu = ops.Split(0)

    # Load dataset
    training_set = Inshop(path=data_dir, pose_file=posefile, resize=256, use_labels=False,
                          max_size=48674, xflip=mirror, batch_size=batch_size)
    batch_num = len(training_set) // batch_size

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    grid_size, images, labels, pose = setup_snapshot_image_grid(dataset=training_set)
    save_image_grid(images, os.path.join(out_dir, 'reals.png'), d_range=[0, 255], size=grid_size)

    # Load infer model
    generator_ema_infer = Generator(z_dim=512, w_dim=512, c_dim=0,
                                    img_resolution=256, img_channels=3, batch_size=images.shape[0],
                                    mapping_kwargs=g_mapping_kwargs, synthesis_kwargs=g_synthesis_kwargs)
    param_dict = load_checkpoint(os.path.join(model_path, 'G_ema.ckpt'))
    load_param_into_net(generator_ema_infer, param_dict)

    grid_z = split_gpu(Tensor(np.random.randn(labels.shape[0], generator_ema_infer.z_dim), ms.float32))
    grid_c = split_gpu(Tensor(labels, ms.float32))
    grid_pose = split_gpu(Tensor(pose, ms.float32))
    imgs = []
    for z, c, pose in zip(grid_z, grid_c, grid_pose):
        ws = generator_ema_infer.mapping.construct(z=z, c=c)
        img = generator_ema_infer.synthesis.construct(ws=ws, pose=pose, noise_mode=0)
        imgs.append(img)
    imgs = concat(imgs).asnumpy()
    save_image_grid(imgs, os.path.join(out_dir, 'fakes_init.png'), d_range=[-1, 1], size=grid_size)

    print('Num images: ', len(training_set), '\nImage shape: ', training_set.image_shape,
          '\nLabel shape:', training_set.label_shape)

    start_time = time.time()

    # Load train model
    generator = Generator(z_dim=512, w_dim=512, c_dim=0,
                          img_resolution=256, img_channels=3, batch_size=batch_size, train=True,
                          mapping_kwargs=g_mapping_kwargs, synthesis_kwargs=g_synthesis_kwargs)
    generator_ema = copy.deepcopy(generator)
    for key, value in generator_ema.parameters_dict().items():
        if 'all_conv' in key:
            value.requires_grad = False
    discriminator = Discriminator(c_dim=0, img_resolution=256, img_channels=3, block_kwargs={},
                                  mapping_kwargs={}, epilogue_kwargs=d_epilogue_kwargs, batch_size=batch_size,
                                  channel_base=16384, channel_max=512, num_fp16_res=4, conv_clamp=256)

    module_list = [('G', generator), ('D', discriminator), ('G_ema', generator_ema)]

    for model_name, module in module_list:
        param_dict = load_checkpoint(os.path.join(model_path, model_name + '.ckpt'))
        load_param_into_net(module, param_dict)

    cal_loss = CustomWithLossCell(generator.mapping, generator.synthesis, discriminator, StyleGANLoss)

    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', generator, g_opt_kwargs, g_reg_interval),
                                                   ('D', discriminator, d_opt_kwargs, d_reg_interval)]:
        mb_ratio = reg_interval / (reg_interval + 1)
        opt_kwargs['lr'] = opt_kwargs['lr'] * mb_ratio
        opt_kwargs['betas'] = [beta ** mb_ratio for beta in opt_kwargs['betas']]
        opt = nn.Adam(module.get_parameters(), learning_rate=Tensor(np.array(opt_kwargs['lr']), ms.float32),
                      beta1=opt_kwargs['betas'][0], beta2=opt_kwargs['betas'][1], eps=opt_kwargs['eps'])
        network = nn.TrainOneStepCell(cal_loss, opt)
        network.set_train()
        phases.append({'name': name + 'main', 'module': module, 'opt': opt, 'interval': 1, 'network': network})

    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    while True:
        for num in range(batch_num):
            (whole_real_img, whole_real_c, whole_pose) = training_set.get_all(num)
            whole_real_img = split_gpu((Tensor(whole_real_img, ms.float32) / 127.5 - 1))
            whole_real_c = split_gpu(Tensor(whole_real_c, ms.float32))
            whole_pose = split_gpu(Tensor(whole_pose, ms.float32))
            whole_gen_z = Tensor(np.random.randn(len(phases) * batch_size, generator.z_dim), ms.float32)
            whole_gen_z = [split_gpu(whole_gen_z) for whole_gen_z in split_batch(whole_gen_z)]
            whole_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in
                           range(len(phases) * batch_size)]
            whole_gen_c = Tensor(np.stack(whole_gen_c), ms.float32)
            whole_gen_c = [split_gpu(whole_gen_c) for whole_gen_c in split_batch(whole_gen_c)]

            for phase, whole_gen_z, whole_gen_c in zip(phases, whole_gen_z, whole_gen_c):
                if batch_idx % phase['interval'] != 0:
                    continue
                phase['module'].requires_grad = True

                # Calculate loss
                for (real_img, pose, real_c, gen_z, gen_c) in \
                        zip(whole_real_img, whole_pose, whole_real_c, whole_gen_z, whole_gen_c):
                    gain = phase['interval']
                    do_gmain = (phase['name'] in ['Gmain', 'Gboth'])
                    do_dmain = (phase['name'] in ['Dmain', 'Dboth'])
                    loss = phase['network'](do_gmain, do_dmain, real_img, pose, real_c, gen_z, gen_c, gain)
                    print('%s loss: %f' % (phase['name'], loss))
                phase['module'].requires_grad = False

            # Update parameters for g_ema
            ema_kimg = 5.0
            ema_nimg = ema_kimg * 1000
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(generator_ema.get_parameters(), generator.get_parameters()):
                value = (p + (p_ema - p) * ema_beta).copy()
                p_ema.set_data(value)

            # Update state
            cur_nimg += batch_size
            batch_idx += 1

            # Perform maintenance tasks once per tick
            done = (cur_nimg >= total_kimg * 1000)
            if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
                continue

            # Print status line, accumulating the same information in stats_collector
            tick_end_time = time.time()
            print('Progress/tick: %5d' % (cur_tick))
            print('Progress/kimg: %8.1f' % (cur_nimg / 1e3))
            print('Timing/total_sec: %12s' % (tick_end_time - start_time))
            print('Timing/sec_per_tick: %7.1f' % (tick_end_time - tick_start_time))
            print(
                'Timing/sec_per_kimg: %7.2f' % ((tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3))
            print('Timing/maintenance_sec: %6.1f' % (maintenance_time))
            print('Timing/total_hours: ', (tick_end_time - start_time) / (60 * 60))
            print('Timing/total_days: ', (tick_end_time - start_time) / (24 * 60 * 60))

            # Check for abort
            if (not done) and (abort_fn is not None):
                done = True
                print('Aborting...')

            # Save network snapshot
            if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
                save_model(module_list, out_dir, cur_nimg)

            # Save image snapshot
            if (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
                save_image_snapshot(out_dir, cur_nimg, generator_ema_infer, grid_z, grid_c, grid_pose,
                                    grid_size, concat)

            # Update state
            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time
            if done:
                all_done = True
                break
        if all_done:
            break
    # Done
    print('Exiting...')


def parse_args():
    """Parameter configuration"""
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--outdir', help='where to save the results', type=str, default='./out/', metavar='DIR')
    parser.add_argument('--gpus', help='number of GPUs to use', type=int, default=1, metavar='INT')
    parser.add_argument('--snap', help='snapshot interval', type=int, default=1, metavar='INT')
    parser.add_argument('--seed', help='random seed', type=int, default=0, metavar='INT')
    parser.add_argument('--data-dir', help='training data', type=str, default='../dataset/data.zip', metavar='PATH')
    parser.add_argument('--posefile', help='csv file of pose keypoints', type=str, default='./pose-annotations.csv')
    parser.add_argument('--mirror', help='enable dataset x-flips', type=bool, default=False, metavar='BOOL')
    parser.add_argument('--total-kimg', help='total training duration', type=int, default=25000, metavar='INT')
    parser.add_argument('--batch-size', help='total batch size', type=int, default=2, metavar='INT')
    parser.add_argument('--need-convert', help='need to convert pkl to ms ckpt', type=bool,
                        default=False, metavar='BOOL')
    parser.add_argument('--model-path', help='path to save models', type=str,
                        default='./ckpt/', metavar='INT')
    parser.add_argument('--device', type=int, default=0, help='device_id')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train(parse_args())
