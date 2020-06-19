import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyntcloud import PyntCloud
from utils import quality_eval
from importlib import import_module

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def metrics_to_dict(metrics, prefix):
    mse, psnr, mae = metrics
    return {f'{prefix}y_mse': mse[0], f'{prefix}u_mse': mse[1], f'{prefix}v_mse': mse[2],
            f'{prefix}y_mae': mae[0], f'{prefix}u_mae': mae[1], f'{prefix}v_mae': mae[2],
            f'{prefix}y_psnr': psnr[0], f'{prefix}u_psnr': psnr[1], f'{prefix}v_psnr': psnr[2]}


def run(file1, file2, point_size=1):
    assert os.path.exists(file1), f'{file1} not found'
    assert os.path.exists(file2), f'{file2} not found'

    file2_folder, _ = os.path.split(file2)
    file2_report = os.path.join(file2_folder, 'report.json')
    assert os.path.exists(file2_report)

    logging.info(f'Updating {file2_report}.')
    with open(file2_report, 'r') as f:
        data = json.load(f)

    pc1 = PyntCloud.from_file(file1)
    pc2 = PyntCloud.from_file(file2)

    cols = ['x', 'y', 'z', 'red', 'green', 'blue']
    final_metrics, fwd_metrics, bwd_metrics = quality_eval.color_with_geo(pc1.points[cols].values, pc2.points[cols].values)

    n_points = len(pc1.points)
    size_in_bytes = data['color_bitstream_size_in_bytes']
    size_in_bits = size_in_bytes * 8
    bpp = size_in_bits / n_points
    data = {**data,
            **metrics_to_dict(final_metrics, ''),
            **metrics_to_dict(fwd_metrics, 'AB_'),
            **metrics_to_dict(bwd_metrics, 'BA_'),
            'color_bits_per_input_point': bpp,
            'input_point_count': n_points}
    with open(file2_report, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)
    logging.info(f'{file2_report} written.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='23_eval_merged.py', description='Eval a merged point cloud.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('file1', help='Original file.')
    parser.add_argument('file2', help='Distorted file.')
    parser.add_argument('--point_size', default=1, type=int)
    args = parser.parse_args()

    run(args.file1, args.file2, args.point_size)
