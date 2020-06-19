import numpy as np
import multiprocessing as mp
from tqdm import trange
from numba import njit
from scipy.spatial import cKDTree


@njit
def compute_pt_curvature(neighbors_pts):
    n_neighbors = neighbors_pts.shape[0]
    cov = np.cov(neighbors_pts, rowvar=False).astype(np.float32)
    u, s, vh = np.linalg.svd(cov)
    neighbors_pts_trs = neighbors_pts @ vh.T

    x = neighbors_pts_trs[:, 0]
    y = neighbors_pts_trs[:, 1]
    A = np.empty((n_neighbors, 6), dtype=np.float32)
    A[:, 0] = x ** 2
    A[:, 1] = y ** 2
    A[:, 2] = x * y
    A[:, 3] = x
    A[:, 4] = y
    A[:, 5] = 1.0
    B = neighbors_pts_trs[:, 2]

    coeffs = np.linalg.lstsq(A, B)[0]
    fxx, fyy, fxy, fx, fy, delta = coeffs
    q = (1 + fx * fx + fy * fy)
    # last eigenvector (n)
    # delta = delta * vh[:, -1]
    # proj = origin + delta

    # mean curvature
    return 0.5 * ((1 + fx ** 2) * fyy + (1 + fy ** 2) * fxx - 2 * fxy * fx * fy) / (q * np.sqrt(q))


def compute_pc_curvature(neighbors):
    curvatures = np.empty((neighbors.shape[0],), dtype=np.float32)
    for i in trange(neighbors.shape[0]):
        curvatures[i] = compute_pt_curvature(neighbors[i])

    return curvatures


def compute_pc_norm_curvature(pts, k):
    tree = cKDTree(pts[:, :3])
    _, indices = tree.query(pts[:, :3], k=k, n_jobs=-1)

    neighbors = pts[:, :3][indices] - pts[:, np.newaxis, :3]
    curvatures = compute_pc_curvature(neighbors)

    # Assume gaussian distribution of curvatures
    norm_curvatures = curvatures - np.mean(curvatures)
    norm_curvatures = norm_curvatures / (2 * np.std(norm_curvatures))
    norm_curvatures = np.abs(norm_curvatures)
    norm_curvatures = np.clip(norm_curvatures, 0.0, 1.0)

    return norm_curvatures
