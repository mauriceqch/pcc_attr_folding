import os
import subprocess
import logging
import argparse
import yaml
import re
import multiprocessing
from utils.parallel_process import parallel_process

HOME = '/home/quachmau'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def run(input_filepath, output_filepath, vox):
    return subprocess.Popen(['python', '99_pc_to_vg.py', input_filepath, output_filepath, '--vg_size', str(2 ** vox)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='91_ds_expdata.py', description='Downsample experimental data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment_path', help='Experiments file path.')
    parser.add_argument('vox', type=int)
    parser.add_argument('--num_parallel', help='Number of parallel jobs.', default=multiprocessing.cpu_count(), type=int)
    parser.add_argument('--dataset_dir', type=str, default=f'{HOME}/data/datasets/mpeg_pcc')
    args = parser.parse_args()
    assert args.vox > 0
    vox_str = f'{args.vox:02}'

    with open(args.experiment_path, 'r') as f:
        experiments = yaml.load(f.read())

    logger.info('Starting')
    params = []
    for experiment in experiments:
        pc_name, input_pc, input_norm, mpeg_mode = [experiment[x] for x in ['pc_name', 'input_pc', 'input_norm', 'mpeg_mode']]
        input_filepath = os.path.join(args.dataset_dir, input_pc)
        new_input_pc = re.sub(r'vox\d\d', f'vox{vox_str}', os.path.split(input_pc)[1])
        output_filepath = os.path.join(args.dataset_dir, f'vox{vox_str}', new_input_pc)
        params.append((input_filepath, output_filepath, args.vox))
        if input_norm != input_pc:
            input_norm_filepath = os.path.join(args.dataset_dir, input_norm)
            new_input_norm = re.sub(r'vox\d\d', f'vox{vox_str}', os.path.split(input_norm)[1])
            output_norm_filepath = os.path.join(args.dataset_dir, f'vox{vox_str}', new_input_norm)
            params.append((input_norm_filepath, output_norm_filepath, args.vox))

    parallel_process(run, params, args.num_parallel)
    logger.info('Finished')
