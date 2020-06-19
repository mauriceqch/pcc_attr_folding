import os
import glob
import argparse
import logging
import subprocess
import multiprocessing
from utils.parallel_process import parallel_process

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def process(i, o):
    return subprocess.Popen(['python', '99_pc_to_vg.py', i, o, '--vg_size', str(args.vg_size)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='pc_to_vg_batch.py',
        description='Converts point clouds to voxelized point clouds.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input_pattern', help='Input pattern.')
    parser.add_argument('output_dir', help='Output directory.')
    parser.add_argument('--vg_size', help='Voxel grid resolution.', default=64, type=int)
    parser.add_argument('--num_parallel', help='Number of parallel processes.', default=multiprocessing.cpu_count(), type=int)
    args = parser.parse_args()

    input_files = glob.glob(args.input_pattern, recursive=True)
    input_directory = os.path.normpath(os.path.commonprefix([os.path.split(x)[0] for x in input_files]))
    os.makedirs(args.output_dir, exist_ok=True)
    filenames = [x[len(input_directory)+1:] for x in input_files]
    output_files = [os.path.join(args.output_dir, x) for x in filenames]

    assert all(s.startswith(args.output_dir) for s in output_files), 'Error during path processing'

    params = list(zip(input_files, output_files))
    parallel_process(process, params, args.num_parallel)
