import numpy as np
import logging
from numba import njit
from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


@njit
def assign_mapping(x_tilde, idx_bwd, dists_bwd):
    counts = np.zeros((x_tilde.shape[0]), dtype=np.int32)
    final_idx_bwd = np.zeros(idx_bwd.shape[0], dtype=np.int32)
    bwd_len = len(idx_bwd)
    k = len(idx_bwd[0])
    for i in range(bwd_len):
        idx_list = idx_bwd[i]
        dists_list = dists_bwd[i]
        min_cost_neigh_idx = 0
        min_cost = np.inf
        for j in range(k):
            neigh_idx = idx_list[j]
            cost = counts[neigh_idx] * dists_list[j]
            if cost < min_cost:
                min_cost = cost
                # We store the index of the neighbor, not the index of its location in the neighbors list
                min_cost_neigh_idx = neigh_idx
        counts[min_cost_neigh_idx] += 1
        final_idx_bwd[i] = min_cost_neigh_idx

    return final_idx_bwd, counts


def build_mapping(ori_x, x_tilde, k=9):
    assert len(ori_x.shape) == 2, f'ori_x shape should be 2 dimensions: currently {ori_x.shape}'
    assert len(x_tilde.shape) == 2, f'x_tilde shape should be 2 dimensions: currently {x_tilde.shape}'
    tree_bwd = cKDTree(x_tilde, balanced_tree=False)
    dists_bwd, idx_bwd = tree_bwd.query(ori_x, k=k, n_jobs=-1)

    final_idx_bwd, counts = assign_mapping(x_tilde, idx_bwd, dists_bwd)

    return final_idx_bwd, counts


def compute_occupancy(ori_x, x_tilde):
    _, counts = build_mapping(ori_x, x_tilde)

    return counts


# Mapping colors from ori_x onto x_tilde using nearest neighbor from ori_x to x_tilde
def map_colors_fwd(ori_x, x_tilde, ori_colors, with_bwd=True):
    counts = np.zeros((x_tilde.shape[0], 1))
    color_sums = np.zeros((x_tilde.shape[0], 3))

    idx_bwd, _ = build_mapping(ori_x, x_tilde)

    for i, idx in enumerate(idx_bwd):
        counts[idx] += 1
        color_sums[idx] += ori_colors[i]

    # Forward mapping fills empty space to make the image smooth
    # This decreases bitrate as black pixels are present otherwise
    if with_bwd:
        tree_fwd = cKDTree(ori_x, balanced_tree=False)
        _, idx_fwd = tree_fwd.query(x_tilde)
        for i, idx in enumerate(idx_fwd):
            if counts[i] == 0:
                counts[i] += 1
                color_sums[i] += ori_colors[idx]

    counts = np.maximum(counts, np.ones(counts.shape))
    colors_tilde = np.round(color_sums / counts)
    return colors_tilde


# Mapping back colors from x_tilde onto ori_x using nearest neighbor from ori_x to x_tilde
def map_colors_bwd(x_tilde, ori_x, colors_tilde):
    ori_colors_tilde = np.zeros((ori_x.shape[0], 3))

    idx_bwd, _ = build_mapping(ori_x, x_tilde)

    for i, idx in enumerate(idx_bwd):
        ori_colors_tilde[i] = colors_tilde[idx]

    return ori_colors_tilde


# Bidirectional mapping
def map_colors(ori_x, x_tilde, ori_colors):
    counts = np.ones(x_tilde.shape[0])
    color_sums = np.zeros((x_tilde.shape[0], 3))

    tree_fwd = cKDTree(ori_x, balanced_tree=False)
    _, idx_fwd = tree_fwd.query(x_tilde)
    colors_tilde = ori_colors[idx_fwd]
    color_sums += colors_tilde

    tree_bwd = cKDTree(x_tilde, balanced_tree=False)
    _, idx_bwd = tree_bwd.query(ori_x)

    unique_idx_bwd = np.unique(idx_bwd)
    for i, idx in enumerate(idx_bwd):
        counts[idx] += 1
        color_sums[idx] += ori_colors[i]
    for i, idx in enumerate(unique_idx_bwd):
        colors_tilde[idx] = np.round(color_sums[idx] / counts[idx])
    return colors_tilde
