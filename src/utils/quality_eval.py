from . import color_space
from scipy.spatial import cKDTree
import numpy as np


def color(ori_colors, dist_colors):
    ori_colors = color_space.rgb_to_yuv(ori_colors)
    dist_colors = color_space.rgb_to_yuv(dist_colors)
    mae = np.mean(np.abs((ori_colors - dist_colors) / 255.), axis=0)
    mse = np.mean(np.square((ori_colors - dist_colors) / 255.), axis=0)
    psnr = -10 * np.log10(mse)

    return mse, psnr, mae


def color_with_geo(ori_pts, dist_pts):
    ori_geo = ori_pts[:, :3]
    dist_geo = dist_pts[:, :3]
    ori_col = ori_pts[:, 3:]
    dist_col = dist_pts[:, 3:]

    fwd_tree = cKDTree(dist_geo, balanced_tree=False)
    _, fwd_idx = fwd_tree.query(ori_geo)
    fwd_colors = dist_col[fwd_idx]
    fwd_metrics = color(fwd_colors, ori_col)

    bwd_tree = cKDTree(ori_geo, balanced_tree=False)
    _, bwd_idx = bwd_tree.query(dist_geo)
    bwd_colors = ori_col[bwd_idx]
    bwd_metrics = color(bwd_colors, dist_col)

    assert len(fwd_metrics) == len(bwd_metrics),\
        f'found len(fwd_metrics) = {len(fwd_metrics)} != len(bwd_metrics) = {len(bwd_metrics)}'

    final_metrics = tuple([np.min((fwd_metrics[i], bwd_metrics[i]), axis=0) for i in range(len(fwd_metrics))])

    return final_metrics, fwd_metrics, bwd_metrics
