import logging
import os
import numpy as np
import argparse
import shutil
import json
import pandas as pd
import utils
import datetime
import matplotlib.pyplot as plt
from utils import pc_io, color_space, quality_eval, adj
from utils.bpg import bpgenc, bpgdec, bpgenc_lossless
from utils.color_mapping import map_colors_fwd, map_colors_bwd, compute_occupancy
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def write_pc(geo, attr, path):
    pc_io.write_pc(np.concatenate((geo, attr), 1), path)


def eval_qp(imgpath, ori_colors, ori_x, qp, x_tilde, filename_without_ext, output_folder_name):
    current_folder, filename = os.path.split(imgpath)
    output_folder = os.path.join(current_folder, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    bpgpath = os.path.join(output_folder, f'{filename}.bpg')
    if qp == 'll':
        bpgenc_lossless(imgpath, bpgpath)
    else:
        bpgenc(imgpath, bpgpath, qp)
    bpgdecpath = bpgpath + '.png'
    bpgdec(bpgpath, bpgdecpath)

    bpgdecimg = Image.open(bpgdecpath)
    bpgdecimg = np.asarray(bpgdecimg)
    # Use grid order
    bpgdecimg = bpgdecimg.reshape((-1, 3))
    ori_colors_tilde = map_colors_bwd(x_tilde, ori_x, bpgdecimg)
    remap_pc_path = os.path.join(output_folder, filename_without_ext + '_remap.ply')
    write_pc(ori_x, ori_colors_tilde, remap_pc_path)

    remap_mse, remap_psnr, remap_mae = quality_eval.color(ori_colors, ori_colors_tilde)
    size_in_bytes = os.stat(bpgpath).st_size
    size_in_bits = size_in_bytes * 8
    n_points = ori_x.shape[0]
    bpp = size_in_bits / n_points

    def format_list(l):
        return ' '.join([f'{x:.2E}' for x in l])

    logger.info(f'[{output_folder_name}] MSE: {format_list(remap_mse)}, '
                + f'PSNR: {format_list(remap_psnr)}, size: {size_in_bytes}B, bpp: {bpp:.2f}')
    data = {'qp': qp,
            'y_mse': remap_mse[0], 'u_mse': remap_mse[1], 'v_mse': remap_mse[2],
            'y_mae': remap_mae[0], 'u_mae': remap_mae[1], 'v_mae': remap_mae[2],
            'y_psnr': remap_psnr[0], 'u_psnr': remap_psnr[1], 'v_psnr': remap_psnr[2],
            'color_bitstream_size_in_bytes': size_in_bytes, 'color_bits_per_input_point': bpp,
            'imgpath': imgpath, 'bpgpath': bpgpath, 'bpgdecpath': bpgdecpath,
            'remap_pc_path': remap_pc_path, 'input_point_count': n_points}
    with open(os.path.join(output_folder, 'report.json'), 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)
    return data


def colors_to_img(output, colors, grid, grid_steps):
    img_arr = np.zeros(np.concatenate((grid_steps, [3])), dtype=np.uint8)
    logger.info(f'Writing image at {output} with colors {colors.shape} and grid {grid_steps} ({grid.shape}), img_arr {img_arr.shape}')
    img_arr[grid[:, 0], grid[:, 1]] = colors.astype(np.uint8)
    im = Image.fromarray(img_arr)
    im.save(output)
    return im


def analyze_occupancy(x_tilde, ori_x, grid_steps):
    occupancy = compute_occupancy(ori_x, x_tilde)
    assert len(grid_steps) == 2
    occupancy = np.reshape(occupancy, (grid_steps[0], grid_steps[1]))

    axis_list = [0, 1]
    axis_max_idx_list = []
    axis_max_list = []
    for axis in axis_list:
        reduction_axis = tuple(set(axis_list) - set([axis]))
        # axis_occupancy = np.mean(occupancy, axis=tuple(set(axis_list) - set([axis])))
        axis_occupancy = np.sum(occupancy, axis=reduction_axis) / np.maximum(np.sum(occupancy != 0, axis=reduction_axis), 1)
        axis_max_idx = np.argmax(axis_occupancy)
        axis_max = axis_occupancy[axis_max_idx]
        axis_max_idx_list.append(axis_max_idx)
        axis_max_list.append(axis_max)

    axis_max_list_idx = np.argmax(axis_max_list)
    axis_max = axis_max_list[axis_max_list_idx]
    axis_max_idx = axis_max_idx_list[axis_max_list_idx]
    axis = axis_list[axis_max_list_idx]

    mean_occupancy = np.sum(occupancy) / np.maximum(np.sum(occupancy != 0), 1)
    return axis_max, axis_max_idx, axis, mean_occupancy


def optimize_grid(x_tilde, ori_x, grid_steps, occupancy_threshold=1, ndim=3, rtol=-1e-6):
    assert rtol < 0, 'rtol cannot be positive'
    x_tilde_opt = np.reshape(x_tilde, (grid_steps[0], grid_steps[1], ndim))
    grid_steps_opt = np.copy(grid_steps)

    # Note: we use the mean of the non zeros occupancies
    axis_max, axis_max_idx, axis, mean_occupancy = analyze_occupancy(np.reshape(x_tilde_opt, (-1, ndim)), ori_x, grid_steps_opt)
    ori_mean_occupancy = mean_occupancy
    ori_axis_max = axis_max
    # Placeholder for first check
    previous_mean_occupancy = 2 * mean_occupancy
    i = 0
    while axis_max > occupancy_threshold and\
            (mean_occupancy - previous_mean_occupancy) / previous_mean_occupancy <= rtol:
        print(f'{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M%SZ")} - Iteration {i}, mean occupancy {mean_occupancy}, max occupancy {axis_max}: {grid_steps} => {grid_steps_opt}', end='\r')
        if axis_max_idx == 0:
            # | mid, after ... |
            mid = np.take(x_tilde_opt, axis_max_idx, axis=axis)
            after = np.take(x_tilde_opt, axis_max_idx + 1, axis=axis)
            between_right = (mid + after) / 2
            between_left = (mid - (after - mid))
            # | mid, between_right, after ... |
            x_tilde_opt = np.insert(x_tilde_opt, axis_max_idx + 1, between_right, axis=axis)
            # Remark: between_left is further from mid than between_right
            # This makes the algorithm non symmetric but avoids an exponential reduction of distances on the side
            # | between_left, mid, between_right, after ... |
            x_tilde_opt = np.insert(x_tilde_opt, axis_max_idx, between_left, axis=axis)
        elif axis_max_idx == grid_steps_opt[axis] - 1:
            # | ... before, mid |
            before = np.take(x_tilde_opt, axis_max_idx - 1, axis=axis)
            mid = np.take(x_tilde_opt, axis_max_idx, axis=axis)
            between_left = (before + mid) / 2
            between_right = (mid + (mid - before))
            # Same remark as for the 0 case
            # | ... before, mid, between_right |
            x_tilde_opt = np.insert(x_tilde_opt, axis_max_idx + 1, between_right, axis=axis)
            # | ... before, between_left, mid, between_right |
            x_tilde_opt = np.insert(x_tilde_opt, axis_max_idx, between_left, axis=axis)
        else:
            # | ... before, mid, after ... |
            before = np.take(x_tilde_opt, axis_max_idx - 1, axis=axis)
            mid = np.take(x_tilde_opt, axis_max_idx, axis=axis)
            after = np.take(x_tilde_opt, axis_max_idx + 1, axis=axis)
            between_left = (before + mid) / 2
            between_right = (mid + after) / 2
            # | ... before, mid, between_right, after ... |
            x_tilde_opt = np.insert(x_tilde_opt, axis_max_idx + 1, between_right, axis=axis)
            # | ... before, between_left, mid, between_right, after ... |
            x_tilde_opt = np.insert(x_tilde_opt, axis_max_idx, between_left, axis=axis)

        grid_steps_opt[axis] += 2
        previous_mean_occupancy = mean_occupancy
        axis_max, axis_max_idx, axis, mean_occupancy = analyze_occupancy(np.reshape(x_tilde_opt, (-1, ndim)), ori_x, grid_steps_opt)
        i += 1

    logger.info(f'Iteration {i}, mean occupancy {ori_mean_occupancy} => {mean_occupancy},' +
                f' max occupancy {ori_axis_max} => {axis_max}: {grid_steps} => {grid_steps_opt}')

    return np.reshape(x_tilde_opt, (-1, ndim)), grid_steps_opt


def eval_folding():
    rows = []

    # Paths and folders
    logger.info(f'Loading folding data for {args.input_dir}')
    folding_data_path = os.path.join(args.input_dir, 'folding_data.npz')
    assert os.path.exists(folding_data_path), f'{folding_data_path} does not exist'
    folding_data = np.load(folding_data_path)
    x_tilde, ori_values, norm_params, grid_steps, origin_path, origin_folder = [
        folding_data[k] for k in ['x_tilde', 'ori_values', 'norm_params', 'grid_steps', 'origin_path', 'origin_folder']]
    origin_path = str(origin_path)
    origin_folder = str(origin_folder)

    filepath, filename = os.path.split(origin_path[len(origin_folder) + 1:])
    filename_without_ext, _ = os.path.splitext(filename)
    current_output_dir = os.path.join(args.input_dir, filepath)
    os.makedirs(current_output_dir, exist_ok=True)
    path_without_ext = os.path.join(current_output_dir, filename_without_ext)

    # Denormalize data
    x_tilde = pc_io.denormalize_points(x_tilde, norm_params)
    ori_x = pc_io.denormalize_points(ori_values[:, :3], norm_params)
    ori_colors = np.round(ori_values[:, 3:] * 255.)

    logger.info(f'Refine reconstruction for {origin_path}')
    # Refine the reconstruction
    grid_steps_2d = grid_steps[:2]
    x_tilde_refined = adj.graph_refining(ori_x, x_tilde, grid_steps_2d)

    # Optimize the refined reconstruction
    logger.info(f'Optimize refined reconstruction for {origin_path}')
    x_tilde_refined_opt, grid_steps_2d_opt = optimize_grid(x_tilde_refined, ori_x, grid_steps_2d)

    # Copy original file
    ori_pc_path = path_without_ext + '_ori.ply'
    shutil.copyfile(origin_path, ori_pc_path)

    # Write folded point cloud with colors
    # Not refined
    colors_tilde = map_colors_fwd(ori_x, x_tilde, ori_colors)
    write_pc(x_tilde, colors_tilde, path_without_ext + '_folded.ply')
    colors_tilde_no_bwd = map_colors_fwd(ori_x, x_tilde, ori_colors, with_bwd=False)
    write_pc(x_tilde, colors_tilde_no_bwd, path_without_ext + '_folded_no_bwd.ply')
    # Refined
    colors_tilde_refined = map_colors_fwd(ori_x, x_tilde_refined, ori_colors)
    write_pc(x_tilde_refined, colors_tilde_refined, path_without_ext + '_folded_refined.ply')
    colors_tilde_refined_no_bwd = map_colors_fwd(ori_x, x_tilde_refined, ori_colors, with_bwd=False)
    write_pc(x_tilde_refined, colors_tilde_refined_no_bwd, path_without_ext + '_folded_refined_no_bwd.ply')
    # Refined optimized
    colors_tilde_refined_opt = map_colors_fwd(ori_x, x_tilde_refined_opt, ori_colors)
    write_pc(x_tilde_refined_opt, colors_tilde_refined_opt, path_without_ext + '_folded_refined_opt.ply')
    colors_tilde_refined_opt_no_bwd = map_colors_fwd(ori_x, x_tilde_refined_opt, ori_colors, with_bwd=False)
    write_pc(x_tilde_refined_opt, colors_tilde_refined_opt_no_bwd, path_without_ext + '_folded_refined_opt_no_bwd.ply')

    # For values in [0, 1]
    def to_psnr(se):
        return -10 * np.log10(np.maximum(se, 1e-8))

    # Write Y absolute errors
    # Not refined
    color_errors = np.abs(
        color_space.rgb_to_yuv(map_colors_bwd(x_tilde, ori_x, colors_tilde))[:, 0]
        - color_space.rgb_to_yuv(ori_colors)[:, 0])
    color_errors = np.repeat(color_errors[:, np.newaxis], 3, axis=1)
    write_pc(ori_x, color_errors, path_without_ext + '_ori_errors.ply')
    color_serrors = np.square(color_errors[:, 0] / 255.)
    fig, ax = plt.subplots()
    ax.hist(to_psnr(color_serrors), bins=100)
    fig.savefig(path_without_ext + '_ori_errors_hist.png')
    fig, ax = plt.subplots()
    ax.hist(to_psnr(color_serrors[color_serrors > 0]), bins=100)
    fig.savefig(path_without_ext + '_ori_errors_hist_nonzero.png')
    # Refined
    color_errors_refined = np.abs(
        color_space.rgb_to_yuv(map_colors_bwd(x_tilde_refined, ori_x, colors_tilde_refined))[:, 0]
        - color_space.rgb_to_yuv(ori_colors)[:, 0])
    color_errors_refined = np.repeat(color_errors_refined[:, np.newaxis], 3, axis=1)
    write_pc(ori_x, color_errors_refined, path_without_ext + '_ori_errors_refined.ply')
    color_serrors_refined = np.square(color_errors_refined[:, 0] / 255.)
    fig2, ax2 = plt.subplots()
    ax2.hist(to_psnr(color_serrors_refined), bins=100)
    fig2.savefig(path_without_ext + '_ori_errors_refined_hist.png')
    fig2, ax2 = plt.subplots()
    ax2.hist(to_psnr(color_serrors_refined[color_serrors_refined > 0]), bins=100)
    fig2.savefig(path_without_ext + '_ori_errors_refined_hist_nonzero.png')
    # Refined optimized
    color_errors_refined_opt = np.abs(
        color_space.rgb_to_yuv(map_colors_bwd(x_tilde_refined_opt, ori_x, colors_tilde_refined_opt))[:, 0]
        - color_space.rgb_to_yuv(ori_colors)[:, 0])
    color_errors_refined_opt = np.repeat(color_errors_refined_opt[:, np.newaxis], 3, axis=1)
    write_pc(ori_x, color_errors_refined_opt, path_without_ext + '_ori_errors_refined_opt.ply')
    color_serrors_refined_opt = np.square(color_errors_refined_opt[:, 0] / 255.)
    fig2, ax2 = plt.subplots()
    ax2.hist(to_psnr(color_serrors_refined_opt), bins=100)
    fig2.savefig(path_without_ext + '_ori_errors_refined_opt_hist.png')
    fig2, ax2 = plt.subplots()
    ax2.hist(to_psnr(color_serrors_refined_opt[color_serrors_refined_opt > 0]), bins=100)
    fig2.savefig(path_without_ext + '_ori_errors_refined_opt_hist_nonzero.png')

    grid_xyz = utils.grid.get_grid(grid_steps)
    grid_xyz = pc_io.denormalize_points(grid_xyz, norm_params)
    grid_values = utils.grid.get_grid_int(grid_steps)
    grid_values_opt = utils.grid.get_grid_int([*grid_steps_2d_opt, 1])

    # Write grid and image with colors
    # Not refined
    write_pc(grid_xyz, colors_tilde, path_without_ext + '_grid.ply')
    imgpath = path_without_ext + '.png'
    colors_to_img(imgpath, colors_tilde, grid_values, grid_steps_2d)
    colors_to_img(path_without_ext + '_no_bwd.png', colors_tilde_no_bwd, grid_values, grid_steps_2d)
    # Refined
    write_pc(grid_xyz, colors_tilde_refined, path_without_ext + '_refined_grid.ply')
    imgpath_refined = path_without_ext + '_refined.png'
    colors_to_img(imgpath_refined, colors_tilde_refined, grid_values, grid_steps_2d)
    colors_to_img(path_without_ext + '_refined_no_bwd.png', colors_tilde_refined_no_bwd, grid_values, grid_steps_2d)
    # Refined optimized
    imgpath_refined_opt = path_without_ext + '_refined_opt.png'
    colors_to_img(imgpath_refined_opt, colors_tilde_refined_opt, grid_values_opt, grid_steps_2d_opt)
    colors_to_img(path_without_ext + '_refined_opt_no_bwd.png', colors_tilde_refined_opt_no_bwd, grid_values_opt, grid_steps_2d_opt)

    # Compress image at various rates
    qps = list(range(20, 55, 5))
    # qps = ['ll'] + list(range(20, 55, 5))
    # Not refined
    for qp in qps:
        result = eval_qp(imgpath, ori_colors, ori_x, qp, x_tilde, filename_without_ext, f'qp_{qp}')
        result = {**result, **{'grid_steps': grid_steps_2d, 'ori_pc_path': ori_pc_path, 'type': 'base'}}
        rows.append(result)
    # Refined
    for qp in qps:
        result = eval_qp(imgpath_refined, ori_colors, ori_x, qp, x_tilde_refined, filename_without_ext,
                         f'refined_qp_{qp}')
        result = {**result, **{'grid_steps': grid_steps_2d, 'ori_pc_path': ori_pc_path, 'type': 'refined'}}
        rows.append(result)
    # Refined optimized
    # Results in higher bitrates as optimization only targets distortion
    for qp in qps:
        result = eval_qp(imgpath_refined_opt, ori_colors, ori_x, qp, x_tilde_refined_opt, filename_without_ext,
                         f'refined_opt_qp_{qp}')
        result = {**result, **{'grid_steps': grid_steps_2d, 'ori_pc_path': ori_pc_path, 'type': 'refined_opt'}}
        rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.input_dir, 'data.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='21_eval_folding.py',
        description='Uses results of gen_folding for evaluation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help='Input directory containing results of gen_folding.')
    args = parser.parse_args()

    os.makedirs(os.path.split(args.input_dir)[0], exist_ok=True)
    eval_folding()
