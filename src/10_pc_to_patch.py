import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.sparse import eye
from pyntcloud import PyntCloud
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from utils.adj import normalize_adj_rows
from utils.curvature import compute_pc_norm_curvature
from utils.pc_io import arr_to_pc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

N_PATCHES_DEFAULT = 9
# K_GRAPH_DEFAULT = 16
# K_CURV_DEFAULT = 64
K_GRAPH_FACTOR_DEFAULT = 0.4
K_CURV_FACTOR_DEFAULT = 1.6
CURV_FACTOR_DEFAULT = 0.04
SMOOTHING_FACTOR_DEFAULT = 1.7


def run(input_file, output_file,
        n_patches=N_PATCHES_DEFAULT,
        k_graph_factor=K_GRAPH_FACTOR_DEFAULT,
        k_curv_factor=K_CURV_FACTOR_DEFAULT,
        smoothing_factor=SMOOTHING_FACTOR_DEFAULT,
        curv_factor=CURV_FACTOR_DEFAULT):
    assert os.path.exists(input_file), f'{input_file} not found'

    pc = PyntCloud.from_file(input_file)
    pts = pc.points.values
    pts_geo = pts[:, :3]
    cols = pc.points.columns
    dtypes = pc.points.dtypes

    if n_patches == 'auto':
        n_patches = max(int(len(pts) / (256 * 256)), 1)
    else:
        n_patches = int(n_patches)
        assert n_patches > 0

    scale = np.linalg.norm(np.max(pts_geo, axis=0) - np.min(pts_geo, axis=0))
    logger.info(f'Processing {input_file} with {len(pts)} points into {n_patches} patches with scale {scale}.')

    pts_cbrt = np.cbrt(len(pts))
    if curv_factor > 0:
        logging.info(f'Computing curvature')
        norm_curvatures = compute_pc_norm_curvature(pts, int(np.round(k_curv_factor * pts_cbrt)))
        scaled_curvatures = norm_curvatures * scale * curv_factor

        pts_geo_final = np.concatenate((pts_geo, scaled_curvatures[:, np.newaxis]), axis=1)
    else:
        pts_geo_final = pts_geo

    if n_patches > 1:
        logging.info(f'Computing kneighbors and clustering')
        knn_graph = kneighbors_graph(pts_geo_final, int(np.round(k_graph_factor * pts_cbrt)), include_self=False, n_jobs=-1)
        clustering = AgglomerativeClustering(
            connectivity=knn_graph, n_clusters=n_patches, affinity='euclidean', linkage='ward')
        labels = clustering.fit_predict(pts_geo_final)

        if smoothing_factor > 0:
            logging.info(f'Smoothing')
            knn_graph_self = knn_graph + eye(*knn_graph.shape)
            knn_graph_self = normalize_adj_rows(knn_graph_self)
            labels_one_hot = np.zeros((len(labels), n_patches))
            labels_one_hot[np.arange(len(labels)), labels] = 1
            # smoothing_iters = int(np.round(0.001 * len(pts)))
            smoothing_iters = int(np.round(2 * pts_cbrt))
            for i in trange(smoothing_iters):
                labels_one_hot = knn_graph_self.dot(labels_one_hot)
                points_per_cluster = np.sum(labels_one_hot, axis=0)
                # labels_one_hot = labels_one_hot / points_per_cluster
                labels_one_hot = labels_one_hot / np.linalg.norm(labels_one_hot, axis=1)[:, np.newaxis]

            final_labels = np.argmax(labels_one_hot, axis=1)
        else:
            final_labels = labels
    else:
        final_labels = np.zeros(pts_geo_final.shape[0], dtype=np.uint32)

    output_folder, _ = os.path.split(output_file)
    os.makedirs(output_folder, exist_ok=True)
    filename, ext = os.path.splitext(output_file)
    for i in range(n_patches):
        current_pts = pts[final_labels == i]

        pc_trs = arr_to_pc(current_pts, cols, dtypes)
        current_output_file = f'{filename}_{i:02}.ply'
        pc_trs.to_file(current_output_file, as_text=True)
        logging.info(f'{current_output_file} written.')

    cmap = plt.get_cmap('gist_ncar')
    colors_seg = np.round(cmap(final_labels / max(n_patches - 1, 1))[:, :3] * 255)
    pts_seg = np.concatenate((pts_geo, colors_seg), axis=1)
    pc_seg = arr_to_pc(pts_seg, cols, dtypes)
    seg_output_file = f'{filename}_seg.ply'
    pc_seg.to_file(seg_output_file, as_text=True)
    logging.info(f'{seg_output_file} written.')

    if curv_factor > 0:
        colors_curv = plt.get_cmap('inferno')(np.round(norm_curvatures * 255).astype(np.uint8), bytes=True)
        pts_curv = np.concatenate((pts_geo, colors_curv[:, :3]), axis=1)
        pc_curv = arr_to_pc(pts_curv, cols, dtypes)
        curv_output_file = f'{filename}_curv.ply'
        pc_curv.to_file(curv_output_file, as_text=True)
        logging.info(f'{curv_output_file} written.')

    logging.info('Finished processing.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='10_pc_to_patch.py', description='Divide a point cloud to patches.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file', help='Input file.')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('--n_patches', default=N_PATCHES_DEFAULT, help='Number of patches (int or \'auto\').')
    parser.add_argument('--k_graph_factor', type=float, default=K_GRAPH_FACTOR_DEFAULT, help='Number of neighbors factor to build the KNN graph.')
    parser.add_argument('--k_curv_factor', type=float, default=K_CURV_FACTOR_DEFAULT, help='Number of neighbors factor to compute curvature.')
    parser.add_argument('--curv_factor', type=float, default=CURV_FACTOR_DEFAULT, help='Curvature factor.')
    parser.add_argument('--smoothing_factor', type=float, default=SMOOTHING_FACTOR_DEFAULT, help='Smoothing factor.')
    args = parser.parse_args()

    run(args.input_file, args.output_file,
        n_patches=args.n_patches,
        k_graph_factor=args.k_graph_factor,
        k_curv_factor=args.k_curv_factor,
        smoothing_factor=args.smoothing_factor,
        curv_factor=args.curv_factor)
