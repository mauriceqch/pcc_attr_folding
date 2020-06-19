import os
import subprocess
import logging
import argparse
import yaml
from tqdm import tqdm

HOME = '/home/quachmau'
MPEG_DATASET_DIR = f'{HOME}/data/datasets/mpeg_pcc'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='72_run_eval_compare_two.py', description='Run eval compare between experiments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment_path', help='Experiments file path.')
    parser.add_argument('mpeg_path', help='Mpeg path.')
    parser.add_argument('mpeg2_path', help='Mpeg path.')
    parser.add_argument('folding_path', help='Folding path.')
    args = parser.parse_args()

    with open(args.experiment_path, 'r') as f:
        experiments = yaml.load(f.read())

    logger.info('Starting')
    for experiment in tqdm(experiments):
        pc_name, input_pc, mpeg_mode = [experiment[x] for x in ['pc_name', 'input_pc', 'mpeg_mode']]
        mpeg_output_dir = os.path.join(args.mpeg_path, mpeg_mode, pc_name)
        mpeg2_output_dir = os.path.join(args.mpeg2_path, mpeg_mode, pc_name)
        folding_output_dir = os.path.join(args.folding_path, pc_name)
        subprocess.run(['python', '73_eval_compare_two.py', mpeg_output_dir, mpeg2_output_dir, folding_output_dir], check=True)
    logger.info('Finished')
