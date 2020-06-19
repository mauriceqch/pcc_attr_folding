import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.curvature import compute_pc_norm_curvature
from utils.pc_io import arr_to_pc
from pyntcloud import PyntCloud

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def run(input_file, output_file, k):
    assert os.path.exists(input_file), f'{input_file} not found'

    pc = PyntCloud.from_file(input_file)
    pts = pc.points.values
    cols = pc.points.columns
    dtypes = pc.points.dtypes

    logger.info(f'Computing curvature for {output_file}')
    norm_curvatures = compute_pc_norm_curvature(pts, k)
    #norm_curvatures = np.floor(norm_curvatures + 0.5)
    colors = plt.get_cmap('inferno')(np.round(norm_curvatures * 255).astype(np.uint8), bytes=True)

    pts[:, 3:6] = colors[:, :3]

    pc_trs = arr_to_pc(pts, cols, dtypes)
    pc_trs.to_file(output_file, as_text=True)
    logging.info(f'Finished processing {output_file}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='994_pc_curvature.py', description='Compute point cloud curvature.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file', help='Input file.')
    parser.add_argument('output_file', help='Output file')
    parser.add_argument('--k', default=128, type=int, help='Number of nearest neighbors')
    args = parser.parse_args()

    run(args.input_file, args.output_file, args.k)
