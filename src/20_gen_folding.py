import logging
import os
import glob
import numpy as np
import tensorflow.compat.v1 as tf
import argparse
import importlib
from tqdm import trange
from utils import grid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


def gen_folding():
    train_test_split = False
    infinite_data = False
    batch_size = 1

    files = glob.glob(args.input_pattern, recursive=True)
    if len(files) > 1:
        origin_folder = os.path.commonpath(files)
    elif '*' in files[0]:
        origin_folder = files[0][:args.input_pattern.find('*') - 1]
    else:
        origin_folder = os.path.split(files[0])[0]
    assert len(files) > 0, "No files found"

    Model = getattr(importlib.import_module(args.model), 'Model')
    InputPipeline = getattr(importlib.import_module(args.input_pipeline), 'InputPipeline')

    rows = []
    input_pipeline = InputPipeline(files, batch_size, train_test_split, infinite_data)

    n = len(input_pipeline.data[0][0])
    grid_steps = grid.parse_grid_steps(args.grid_steps, n) * np.array([args.grid_steps_factor, args.grid_steps_factor, 1])
    grid_steps = grid_steps.astype(np.uint32)
    grid_values = grid.get_grid(grid_steps).astype(np.float32)
    logger.info(f'Grid steps: {grid_steps}')

    model = Model(input_pipeline)

    # Init
    init = tf.global_variables_initializer()

    # Checkpoints
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    with tf.Session() as sess:
        sess.run(init)
        checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        if checkpoint is not None:
            saver.restore(sess, checkpoint)
        else:
            raise Exception(f'no checkpoint found at {args.checkpoint_dir}')

        for i in trange(len(files)):
            logger.info(f'Run network for {files[i]}')
            # Run network
            x_tilde, col_loss = sess.run([model.x_tilde, model.col_loss], feed_dict={model.grid: grid_values})
            logger.info(f'grid: {grid_steps}, col_loss: {col_loss}')

            # Obtain data
            ori_values, norm_params = input_pipeline.data[i]

            # Paths and folders
            origin_path = files[i]
            filepath, filename = os.path.split(origin_path[len(origin_folder) + 1:])
            current_output_dir = os.path.join(args.output_dir, filepath)
            os.makedirs(current_output_dir, exist_ok=True)
            output_file = os.path.join(current_output_dir, 'folding_data')

            np.savez_compressed(output_file,
                                x_tilde=x_tilde, ori_values=ori_values, norm_params=norm_params,
                                grid_steps=grid_steps, origin_path=origin_path, origin_folder=origin_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='20_gen_folding.py',
        description='Compute the folding using a trained model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_pattern', help='Input pattern.')
    parser.add_argument('output_dir', help='Output directory.')
    parser.add_argument('checkpoint_dir', help='Directory where to save/load model checkpoints.')
    parser.add_argument('--model', help='Model module.', required=True)
    parser.add_argument('--input_pipeline', help='Input pipeline module.', required=True)
    parser.add_argument('--filters', type=int, default=32, help='Number of filters per layer.')
    parser.add_argument('--max_steps', type=int, default=100000, help='Train up to this number of steps.')
    parser.add_argument('--grid_steps', default='512,512,1')
    parser.add_argument('--grid_steps_factor', default=1.0, type=float, help='Multiply grid_steps by this factor.')
    args = parser.parse_args()

    os.makedirs(os.path.split(args.checkpoint_dir)[0], exist_ok=True)
    os.makedirs(os.path.split(args.output_dir)[0], exist_ok=True)
    gen_folding()
