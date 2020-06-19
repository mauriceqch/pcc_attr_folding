import pyntcloud
import argparse
import logging
import numpy as np
import os
from utils import pc_io, grid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def run(input, output):
    logger.info(f'Starting {input} to {output}')
    pc = pyntcloud.PyntCloud.from_file(input)
    pts = pc.points.values

    n = int(np.sqrt(pts.shape[0]))
    assert n*n == pts.shape[0]

    t = grid.grid_borders_mask((n, n))
    pts[t, 3:] = 255.0

    output_folder, _ = os.path.split(output)
    os.makedirs(output_folder, exist_ok=True)
    pc_io.write_pc(pts, output)
    logger.info(f'Finished {input} to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='98_highlight_borders.py',
        description='Highlight the grid borders for a folded point cloud.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input', help='Input file.')
    parser.add_argument('output', help='Output file.')
    args = parser.parse_args()

    run(args.input, args.output)
