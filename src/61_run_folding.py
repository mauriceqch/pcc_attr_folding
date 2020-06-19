import os
import subprocess
import logging
import argparse
import yaml
from tqdm import tqdm

HOME = '/home/quachmau'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='61_run_folding.py', description='Run folding experiments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment_path', help='Experiments file path.')
    parser.add_argument('output_path', help='Output path.')
    parser.add_argument('--dataset_dir', type=str, default=f'{HOME}/data/datasets/mpeg_pcc')
    args = parser.parse_args()

    with open(args.experiment_path, 'r') as f:
        experiments = yaml.load(f.read())

    logger.info('Starting Folding experiments')
    for experiment in tqdm(experiments):
        pc_name, input_pc, point_size = [experiment[x] for x in ['pc_name', 'input_pc', 'point_size']]
        current_output_path = os.path.join(args.output_path, pc_name)
        subprocess.run(['python', '60_folding_pipeline.py',
                        os.path.join(args.dataset_dir, input_pc),
                        current_output_path,
                        '--point_size', str(point_size)], check=True)
    logger.info('Finished Folding experiments')
