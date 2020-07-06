import argparse
import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from utils import bd
from glob import glob

rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 12
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.figsize'] = 7.3, 4.2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def read_json(file):
    with open(file, 'r') as f:
        return json.load(f)


def build_curves(mpeg_data, cmp_data, refined_cmp_data, refined_opt_cmp_data, ylabel, filename, cmp_path, ylim=None, xlim=None, legend_loc='lower right'):
    logger.info(f'Building curves with {ylabel}')
    logger.debug(f'mpeg_data: {mpeg_data}')
    logger.debug(f'cmp_data: {cmp_data}')

    mpeg_finite = np.isfinite(mpeg_data[:, 1])

    if not np.all(mpeg_finite):
        mpeg_lossless = mpeg_data[~mpeg_finite]
        mpeg_lossless_bpp = np.min(mpeg_lossless[:, 0])

    mpeg_data = mpeg_data[mpeg_finite]
    cmp_data = cmp_data[np.isfinite(cmp_data[:, 1])]
    refined_cmp_data = refined_cmp_data[np.isfinite(refined_cmp_data[:, 1])]
    refined_opt_cmp_data = refined_opt_cmp_data[np.isfinite(refined_opt_cmp_data[:, 1])]

    fig, ax = plt.subplots()
    ax.plot(mpeg_data[:, 0], mpeg_data[:, 1], label='GPCC', linestyle='-', marker='o')
    if not np.all(mpeg_finite):
        ax.axvline(x=mpeg_lossless_bpp, label='GPCC (lossless)', linestyle='-')
    ax.plot(cmp_data[:, 0], cmp_data[:, 1], label='Folding', linestyle='--', marker='s')
    ax.plot(refined_cmp_data[:, 0], refined_cmp_data[:, 1], label='Refined folding', linestyle='-.', marker='v')
    ax.plot(refined_opt_cmp_data[:, 0], refined_opt_cmp_data[:, 1], label='Opt. Refined folding', linestyle='-.', marker='x')
    ax.set(xlabel='bits per input point', ylabel=ylabel)
    ax.set_xlim(left=0)
    ax.legend(loc=legend_loc)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    ax.grid(True)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    for ext in ['.pdf', '.png']:
        fig.savefig(os.path.join(cmp_path, filename + ext))

    bdrate = bd.bdrate(mpeg_data, cmp_data)
    refined_bdrate = bd.bdrate(mpeg_data, refined_cmp_data)
    refined_opt_bdrate = bd.bdrate(mpeg_data, refined_opt_cmp_data)
    message = f'bdrate {ylabel}: {bdrate}\n' +\
              f'refined bdrate {ylabel}: {refined_bdrate}\n' +\
              f'refined opt bdrate {ylabel}: {refined_opt_bdrate}\n'
    logger.info(message)
    with open(os.path.join(cmp_path, 'eval.log'), 'a') as f:
        f.write(message + '\n')


def run(mpeg_path, cmp_path):
    assert os.path.exists(mpeg_path), f'{mpeg_path} does not exist'
    assert os.path.exists(cmp_path), f'{cmp_path} does not exist'

    mpeg_reports = glob(os.path.join(mpeg_path, '**/report.json'), recursive=True)
    cmp_reports = glob(os.path.join(cmp_path, 'merged/**/report.json'), recursive=True)
    refined_cmp_reports = glob(os.path.join(cmp_path, 'refined_merged/**/report.json'), recursive=True)
    refined_opt_cmp_reports = glob(os.path.join(cmp_path, 'refined_opt_merged/**/report.json'), recursive=True)
    assert len(mpeg_reports) > 0
    assert len(cmp_reports) > 0
    assert len(refined_cmp_reports) > 0
    assert len(refined_opt_cmp_reports) > 0
    logger.info(f'mpeg_reports: {mpeg_reports}')
    logger.info(f'cmp_reports: {cmp_reports}')
    logger.info(f'refined_cmp_reports: {refined_cmp_reports}')
    logger.info(f'refined_opt_cmp_reports: {refined_opt_cmp_reports}')

    mpeg_reports = [read_json(x) for x in mpeg_reports]
    cmp_reports = [read_json(x) for x in cmp_reports]
    refined_cmp_reports = [read_json(x) for x in refined_cmp_reports]
    refined_opt_cmp_reports = [read_json(x) for x in refined_opt_cmp_reports]

    mpeg_df = pd.DataFrame(data=mpeg_reports)
    cmp_df = pd.DataFrame(data=cmp_reports)
    refined_cmp_df = pd.DataFrame(data=refined_cmp_reports)
    refined_opt_cmp_df = pd.DataFrame(data=refined_opt_cmp_reports)

    try:
        os.remove(os.path.join(cmp_path, 'eval.log'))
    except OSError:
        pass

    curves = [
        ('rd_curve_y_psnr', 'y_psnr', 'Y PSNR (dB)', 'lower right'),
        # ('rd_curve_y_mae', 'y_mae', 'Y MAE (dB)', 'upper right'),
        ('rd_curve_y_mse', 'y_mse', 'Y MSE (dB)', 'upper right')
    ]

    for (filename, column, ylabel, legend_loc) in curves:
        build_curves(
            mpeg_df[['color_bits_per_input_point', column]].values,
            cmp_df[['color_bits_per_input_point', column]].values,
            refined_cmp_df[['color_bits_per_input_point', column]].values,
            refined_opt_cmp_df[['color_bits_per_input_point', column]].values,
            ylabel, filename, cmp_path, legend_loc=legend_loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='71_eval_compare.py', description='Gathers reports and produces summary.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mpeg_path', help='MPEG test folder.')
    parser.add_argument('cmp_path', help='Comparison method test folder.')
    args = parser.parse_args()

    run(args.mpeg_path, args.cmp_path)
