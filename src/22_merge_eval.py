import os
import argparse
import logging
import numpy as np
import pandas as pd
import json
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

    data = []
    for f in tqdm(input_files):
        with open(f, 'r') as fh:
            d = json.load(fh)
            data.append(d)

    assert len(set([x['qp'] for x in data])) == 1
    merged_data = {
        'qp': data[0]['qp'],
        'color_bitstream_size_in_bytes': sum(x['color_bitstream_size_in_bytes'] for x in data),
        'data': data
    }

    output_folder, _ = os.path.split(output_file)
    os.makedirs(output_folder, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, sort_keys=True, indent=4)
    logging.info(f'{output_file} written.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='22_merge_eval.py', description='Merge evaluation results at different QPs into one file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_files', help='Input files.', nargs='+')
    parser.add_argument('output_file', help='Output file')
    args = parser.parse_args()

    run(args.input_files, args.output_file)
