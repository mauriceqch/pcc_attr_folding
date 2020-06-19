import numpy as np
import logging
import sys
from .grid import grid_borders_mask
from sklearn.feature_extraction.image import grid_to_graph
from scipy.sparse import spdiags, coo_matrix, eye, bmat, csr_matrix
from scipy.spatial import cKDTree
from tqdm import trange

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s:%(lineno)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


# Adj from p1 to p2
# For each point in p2, find its nearest neighbors in p1
# p1: N1 x d
# p2: N2 x d
# returns an N2 x N1 adj matrix
def compute_adj(p1, p2, k=1):
    tree = cKDTree(p1, balanced_tree=False)
    # d: N2 x k, distances
    # i: N2 x k with values from 0 to N1 - 1, col indices
    d, i = tree.query(p2, k, n_jobs=-1)
    i = i.reshape(-1)
    # d = np.ones(i.shape)
    d = d.reshape(-1)
    # j: N2 x k, row indices from 0 to N2 - 1
    j = np.repeat(range(len(p2)), k)

    return coo_matrix((d, (j, i)), shape=(len(p2), len(p1)))


def normalize_adj_rows(adj):
    n, m = adj.shape
    diags = np.array(adj.sum(axis=1))
    diags = np.divide(1, diags, out=np.zeros_like(diags), where=np.abs(diags) > 1e-16)
    diags_tilde = spdiags(diags.flatten(), [0], n, n, format='coo')
    adj_tilde_norm = diags_tilde * adj

    return adj_tilde_norm


def compute_inverse_density(grid_adj, pts_folded):
    grid_dists = np.sum(np.square(pts_folded[grid_adj.row] - pts_folded[grid_adj.col]), axis=1)
    grid_dists = grid_dists / np.array(grid_adj.sum(axis=1)).flatten()[grid_adj.row]
    grid_adj_dists = coo_matrix(grid_adj)
    grid_adj_dists.data = grid_dists
    # Estimate inverse density and weight values using this estimate
    v = np.array(grid_adj_dists.sum(axis=0)).flatten()
    v = v - np.min(v)
    v = v / np.max(v)
    v_diags = spdiags(v, [0], grid_adj.shape[1], grid_adj.shape[1])
    return grid_adj_dists * v_diags


def build_adj(grid_adj, pts_ori, pts_folded):
    total_points = pts_ori.shape[0] + pts_folded.shape[0]

    # inverse density
    grid_adj_id = compute_inverse_density(grid_adj, pts_folded)
    grid_adj_id_full = normalize_adj_rows(grid_adj_id)
    grid_adj_id_full.resize(total_points, total_points)

    # adj_AB = compute_adj(pts_ori, pts_folded)
    # adj_AB = compute_adj(pts_folded, pts_ori).transpose()
    adj_AB_fwd = compute_adj(pts_ori, pts_folded)
    adj_AB_bwd = compute_adj(pts_folded, pts_ori).transpose()
    adj_AB = adj_AB_fwd + adj_AB_bwd
    adj_AB_data = adj_AB.data
    adj_AB_data = adj_AB_data - np.min(adj_AB_data)
    adj_AB_data = adj_AB_data / np.max(adj_AB_data)
    adj_AB.data = adj_AB_data
    # adj_AB = csr_matrix((pts_folded.shape[0], pts_ori.shape[0]))
    adj_BA = csr_matrix((pts_ori.shape[0], pts_folded.shape[0]))
    adj_AA = eye(pts_ori.shape[0])
    # adj_BB = eye(pts_folded.shape[0])
    adj_BB = csr_matrix((pts_folded.shape[0], pts_folded.shape[0]))
    adj_full = bmat([[adj_BB, adj_AB],
                     [adj_BA, adj_AA]], format='csr')
    adj_full = normalize_adj_rows(adj_full)

    final_adj = grid_adj_id_full + 2*adj_full
    final_adj = normalize_adj_rows(final_adj)

    return final_adj


def graph_refining(pts_ori, pts_folded, grid_steps, iters=100):
    grid_adj = grid_to_graph(grid_steps[0], grid_steps[1])
    grid_adj = grid_adj - eye(grid_adj.shape[0], grid_adj.shape[1])
    grid_adj = grid_adj.tocoo()

    borders_mask = grid_borders_mask(grid_steps).reshape((-1,))
    borders_mask_padded = np.concatenate((borders_mask, np.zeros(pts_ori.shape[0]) == 1), 0)

    final_adj = build_adj(grid_adj, pts_ori, pts_folded)

    all_pts = np.concatenate((pts_folded, pts_ori), 0)
    new_pts_folded = all_pts
    for i in trange(iters):
        new_pts_folded = final_adj.dot(new_pts_folded)
        # new_pts_folded = grid_adj_id_full.dot(new_pts_folded)
        # new_pts_folded[borders_mask_padded] = pts_folded[borders_mask]
        final_adj = build_adj(grid_adj, new_pts_folded[pts_folded.shape[0]:], new_pts_folded[:pts_folded.shape[0]])

    return new_pts_folded[:pts_folded.shape[0]]
