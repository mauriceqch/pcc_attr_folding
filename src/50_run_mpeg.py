import os
import subprocess
import logging
import argparse
import yaml
import multiprocessing
from utils.parallel_process import parallel_process
from pyntcloud import PyntCloud

# Config
HOME = '/home/quachmau'
MPEG_TMC13_DIR = f'{HOME}/code/MPEG/mpeg-pcc-tmc13'
TMC13 = f'{MPEG_TMC13_DIR}/build/tmc3/tmc3'
PCERROR = f'{HOME}/code/MPEG/mpeg-pcc-dmetric/test/pc_error_d'
MPEG_DATASET_DIR = f'{HOME}/data/datasets/mpeg_pcc'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def assert_exists(filepath):
    assert os.path.exists(filepath), f'{filepath} not found'


def get_n_points(f):
    return len(PyntCloud.from_file(f).points)


def run_mpeg_experiment(current_mpeg_output_dir, mpeg_cfg_path, input_pc, input_norm):
    input_pc_full = os.path.join(MPEG_DATASET_DIR, input_pc)
    input_norm_full = os.path.join(MPEG_DATASET_DIR, input_norm)
    os.makedirs(current_mpeg_output_dir, exist_ok=True)

    assert_exists(input_pc_full)
    assert_exists(input_norm_full)
    assert_exists(mpeg_cfg_path)

    return subprocess.Popen(['make',
                             '-f', f'{MPEG_TMC13_DIR}/scripts/Makefile.tmc13-step',
                             '-C', current_mpeg_output_dir,
                             f'VPATH={mpeg_cfg_path}',
                             f'ENCODER={TMC13}',
                             f'DECODER={TMC13}',
                             f'PCERROR={PCERROR}',
                             f'SRCSEQ={input_pc_full}',
                             f'NORMSEQ={input_norm_full}'])


def run_gen_report(folder_path, point_size):
    return subprocess.Popen(['python', '51_gen_report.py', folder_path, '--point_size', str(point_size)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='50_run_mpeg.py', description='Run MPEG experiments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment_path', help='Experiments file path.')
    parser.add_argument('output_path', help='Output path.')
    parser.add_argument('--num_parallel', help='Number of parallel jobs.', default=multiprocessing.cpu_count(), type=int)
    args = parser.parse_args()

    assert_exists(TMC13)
    assert_exists(PCERROR)
    assert_exists(MPEG_DATASET_DIR)

    with open(args.experiment_path, 'r') as f:
        experiments = yaml.load(f.read())

    logger.info('Starting GPCC experiments')
    params = []
    for experiment in experiments:
        pc_name, cfg_name, mpeg_mode, input_pc, input_norm = \
            [experiment[x] for x in ['pc_name', 'cfg_name', 'mpeg_mode', 'input_pc', 'input_norm']]
        mpeg_output_dir = os.path.join(args.output_path, mpeg_mode, pc_name)
        for rate in range(1, 7):
            formatted_rate = f'r{rate:02}'
            current_mpeg_output_dir = os.path.join(mpeg_output_dir, formatted_rate)
            mpeg_cfg_path = f'{MPEG_TMC13_DIR}/cfg/{mpeg_mode}/{cfg_name}/{formatted_rate}'
            params.append((current_mpeg_output_dir, mpeg_cfg_path, input_pc, input_norm))
    logger.info('Started GPCC experiments')
    # Issues when running in parallel
    parallel_process(run_mpeg_experiment, params, 1)

    logger.info('Finished GPCC experiments')

    logger.info('Generating GPCC experimental reports')
    params = []
    for experiment in experiments:
        pc_name, mpeg_mode, point_size = [experiment[x] for x in ['pc_name', 'mpeg_mode', 'point_size']]
        mpeg_output_dir = os.path.join(args.output_path, mpeg_mode, pc_name)
        for rate in range(1, 7):
            formatted_rate = f'r{rate:02}'
            current_mpeg_output_dir = os.path.join(mpeg_output_dir, formatted_rate)
            params.append((current_mpeg_output_dir, point_size))
    logger.info('Generated GPCC experimental reports')
    parallel_process(run_gen_report, params, args.num_parallel)

    logger.info('Completing evaluation with image metrics')
