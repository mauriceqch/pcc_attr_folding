import logging
import os
import argparse
import subprocess
import shutil
import multiprocessing
import re
from utils.parallel_process import parallel_process
from pprint import pformat
from glob import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def merge_and_eval(current_output_file, remapped_patches,
                   current_output_report, remapped_reports,
                   input_file, point_size):
    command = []
    if not os.path.exists(current_output_file):
        command.append(' '.join(['python', '12_merge_ply.py', *remapped_patches, current_output_file]))
    if not os.path.exists(current_output_report):
        command.append(' '.join(['python', '22_merge_eval.py', *remapped_reports, current_output_report]))
        command.append(' '.join(['python', '23_eval_merged.py',
                                 input_file, current_output_file,
                                 '--point_size', str(point_size)]))
    return subprocess.Popen(' && '.join(command), shell=True)


def eval_folding(eval_folder):
    return subprocess.Popen(['python', '21_eval_folding.py', eval_folder])


def run(input_file, output_folder, k='auto', max_steps=10000, point_size=1, num_parallel=multiprocessing.cpu_count()):
    filename = os.path.split(input_file)[1]
    filename_without_ext, ext = os.path.splitext(filename)
    model_folder = os.path.join(output_folder, 'model')
    eval_folder = os.path.join(output_folder, 'eval')
    patches_folder = os.path.join(output_folder, 'patches')
    patches_output = os.path.join(patches_folder, filename)

    ori_path = os.path.join(output_folder, filename)
    if not os.path.exists(ori_path):
        shutil.copy(input_file, ori_path)

    logger.info('Creating patches')
    if not os.path.exists(patches_folder):
        subprocess.run(['python', '10_pc_to_patch.py', input_file, patches_output, '--n_patches', str(k)], check=True)
    regexp = re.compile(r'_\d\d\.ply$')
    k = len(list(filter(lambda x: regexp.search(x), glob(os.path.join(patches_folder, '*.ply')))))
    assert k > 0

    logger.info('Init metadata')
    patches_data = {}
    for i in range(k):
        i_str = f'{i:02}'
        current_model_folder = os.path.join(model_folder, i_str)
        current_eval_folder = os.path.join(eval_folder, i_str)
        current_input_file = os.path.join(patches_folder, f'{filename_without_ext}_{i_str}.ply')
        patches_data[i] = {
            'model_folder': current_model_folder,
            'input_file': current_input_file,
            'eval_folder': current_eval_folder,
        }
    logger.info("\n" + pformat(patches_data))

    logger.info('Fitting point clouds')
    for i in range(k):
        pdata = patches_data[i]
        if not os.path.exists(pdata['model_folder']):
            subprocess.run(['python', '11_train.py',
                            pdata['input_file'], pdata['model_folder'],
                            '--max_steps', str(max_steps),
                            '--grid_steps', 'auto', '--model', '80_model', '--input_pipeline', '80_input',
                            ], check=True)

    logger.info('Generate foldings')
    for i in range(k):
        pdata = patches_data[i]
        if not os.path.exists(os.path.join(pdata['eval_folder'], 'folding_data.npz')):
            subprocess.run(['python', '20_gen_folding.py', pdata['input_file'], pdata['eval_folder'], pdata['model_folder'],
                            '--grid_steps', 'auto', '--model', '80_model', '--input_pipeline', '80_input'], check=True)

    logger.info('Eval point cloud attribute compression')
    params = []
    for i in range(k):
        pdata = patches_data[i]
        if not os.path.exists(os.path.join(pdata['eval_folder'], 'data.csv')):
            params.append((pdata['eval_folder'], ))
    parallel_process(eval_folding, params, num_parallel)

    logger.info('Merge patches and eval resulting point cloud')
    params = []
    for prefix in ['', 'refined_', 'refined_opt_']:
        qp_range = range(20, 55, 5)
        for qp in qp_range:
            qp_str = f'{prefix}qp_{qp:02}'
            merged_folder = os.path.join(output_folder, prefix + 'merged')
            current_output_folder = os.path.join(merged_folder, qp_str)
            current_output_file = os.path.join(current_output_folder, filename)
            current_output_report = os.path.join(current_output_folder, 'report.json')

            remapped_patches = glob(os.path.join(eval_folder, '*', qp_str, '*_remap.ply'))
            remapped_reports = glob(os.path.join(eval_folder, '*', qp_str, 'report.json'))
            assert len(remapped_patches) == k, f'Found {len(remapped_patches)} instead of {k} patches for qp {qp}'

            params.append((current_output_file, remapped_patches,
                           current_output_report, remapped_reports,
                           input_file, point_size))
    parallel_process(merge_and_eval, params, num_parallel)

    logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='60_folding_pipeline.py',
        description='Execute the full compression pipeline for a point cloud.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file', help='Input file.')
    parser.add_argument('output_folder', help='Input file.')
    parser.add_argument('--k', help='Number of patches.', default='auto')
    parser.add_argument('--max_steps', help='Number of max steps for each training.', default=10000, type=int)
    parser.add_argument('--point_size', default=1, type=int)
    parser.add_argument('--num_parallel', help='Number of parallel jobs.', default=multiprocessing.cpu_count(), type=int)
    args = parser.parse_args()
    if args.k == 'auto':
        k = 'auto'
    else:
        k = int(args.k)
        assert k >= 1

    os.makedirs(args.output_folder, exist_ok=True)
    run(args.input_file, args.output_folder, k, args.max_steps, args.point_size, args.num_parallel)
