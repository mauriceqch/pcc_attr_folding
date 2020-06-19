import pyntcloud
import argparse
import logging
import numpy as np
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def pc_to_vg(pc, vg_size):
    coords = ['x', 'y', 'z']
    colors = ['red', 'green', 'blue']
    points = pc.points[coords].values
    points = points - np.min(points)
    points = points / np.max(points)
    points = points * (vg_size - 1)
    points = np.round(points)
    pc.points[coords] = points
    if len(set(pc.points.columns) - set(coords)) > 0:
        pc.points = pc.points.groupby(by=coords, sort=False, as_index=False).mean()
        if all(x in set(pc.points.columns) for x in colors):
            pc.points[colors] = pc.points[colors].astype('uint8')
    else:
        pc.points = pc.points.drop_duplicates()

    return pc


def run(input, output, vg_size):
    logger.info(f'Starting {input} to {output} with vg_size {vg_size}')
    pc = pyntcloud.PyntCloud.from_file(input)
    pc = pc_to_vg(pc, vg_size)

    output_folder, _ = os.path.split(output)
    os.makedirs(output_folder, exist_ok=True)
    pc.to_file(output, as_text=True)
    logger.info(f'Finished {input} to {output} with vg_size {vg_size}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='99_pc_to_vg.py',
        description='Transforms a ply point cloud so that it fits on a voxel grid.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input', help='Input file.')
    parser.add_argument('output', help='Output file.')
    parser.add_argument('--vg_size', help='Voxel grid resolution.', default=64, type=int)
    args = parser.parse_args()

    run(args.input, args.output, args.vg_size)
