import os
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyntcloud import PyntCloud

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def run(input_files, output_file):
    for f in input_files:
        assert os.path.exists(f), f'{f} not found'

    logger.info(input_files)
    logger.info(output_file)

    frames = []
    for f in tqdm(input_files):
        pc = PyntCloud.from_file(f)
        frames.append(pc.points)
    final_df = pd.concat(frames)

    output_folder, _ = os.path.split(output_file)
    os.makedirs(output_folder, exist_ok=True)
    PyntCloud(final_df).to_file(output_file)
    logging.info(f'{output_file} written.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='12_merge_ply.py', description='Divide a point cloud to patches.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_files', help='Input file.', nargs='+')
    parser.add_argument('output_file', help='Output file')
    args = parser.parse_args()

    run(args.input_files, args.output_file)
