import logging
import os
import glob
import numpy as np
import tensorflow.compat.v1 as tf
import argparse
import importlib
from tqdm import trange, tqdm
from utils import grid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def train():
    train_test_split = False
    infinite_data = True
    batch_size = 1

    files = glob.glob(args.input_pattern, recursive=True)
    assert len(files) == 1

    Model = getattr(importlib.import_module(args.model), 'Model')
    InputPipeline = getattr(importlib.import_module(args.input_pipeline), 'InputPipeline')

    input_pipeline = InputPipeline(files, batch_size, train_test_split, infinite_data)

    n = len(input_pipeline.data[0][0])
    grid_steps = grid.parse_grid_steps(args.grid_steps, n)
    grid_values = grid.get_grid(grid_steps).astype(np.float32)
    logger.info(f'Grid steps: {grid_steps}')
    model = Model(input_pipeline)
    merged_summary = tf.summary.merge_all()
    no_op = tf.no_op()

    # Summary writers
    train_writer = tf.summary.FileWriter(args.checkpoint_dir)

    # Init
    init = tf.global_variables_initializer()

    # Checkpoints
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model.ckpt')

    # Profiling
    if args.profiling:
        builder = tf.profiler.ProfileOptionBuilder
        opts = builder(builder.time_and_memory()).order_by('micros').build()

    logger.info('Starting session')
    with tf.contrib.tfprof.ProfileContext('./profiler', trace_steps=[], dump_steps=[]) if args.profiling else dummy_context_mgr() as pctx:
        with tf.Session() as sess:
            logger.info('Init session')
            sess.run(init)

            checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
            if checkpoint is not None:
                saver.restore(sess, checkpoint)
            train_writer.add_graph(sess.graph)

            step_val = sess.run(model.step)
            first_step_val = step_val
            pbar = tqdm(total=args.max_steps)
            logger.info(f'Starting training with {files[0]}')
            best_loss = float('inf')
            best_loss_step = step_val
            while step_val <= args.max_steps:
                pbar.update(step_val - pbar.n)

                if args.profiling:
                    pctx.trace_next_step()
                    pctx.dump_next_step()

                # Training
                save_interval = 1000
                summary_interval = 25
                save_model = step_val != first_step_val and step_val % save_interval == 0
                get_summary = step_val % summary_interval == 0 or save_model
                if get_summary:
                    sess_args = [merged_summary, model.train_op, model.rec_loss, model.rep_loss, model.col_loss,
                                 model.col_losses, model.loss]
                else:
                    sess_args = [no_op, model.train_op]

                sess_output = sess.run(sess_args, feed_dict={model.grid: grid_values})

                if args.profiling:
                    pctx.profiler.profile_operations(options=opts)

                step_val = sess.run(model.step)
                if get_summary:
                    train_writer.add_summary(sess_output[0], step_val)
                    pbar.set_description(f"rec_loss: {sess_output[2]:.3E}, rep_loss: {sess_output[3]:.3E},"
                                         + f" col_loss (no_grad): {sess_output[4]:.3E}"
                                         + f" ({', '.join([f'{x:.3E}' for x in sess_output[5]])})")
                    # Early stopping
                    loss = sess_output[6]
                    if loss < best_loss:
                        best_loss_step = step_val
                        best_loss = loss
                    elif step_val - best_loss_step >= 250:
                        save_path = saver.save(sess, checkpoint_path, global_step=step_val)
                        logger.info(f'Early stopping: model saved to {save_path}')
                        break

                if save_model:
                    save_path = saver.save(sess, checkpoint_path, global_step=step_val)
                    logger.info(f'Model saved to {save_path}')
                    



            pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='11_train.py',
        description='Train network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_pattern', help='Input pattern.')
    parser.add_argument('checkpoint_dir', help='Directory where to save/load model checkpoints.')
    parser.add_argument('--model', help='Model module.', required=True)
    parser.add_argument('--input_pipeline', help='Input pipeline module.', required=True)
    parser.add_argument('--max_steps', type=int, default=100000, help='Train up to this number of steps.')
    parser.add_argument('--profiling', default=False, action='store_true', help='Enable profiling')
    parser.add_argument('--grid_steps', default='512,512,1')
    args = parser.parse_args()

    os.makedirs(os.path.split(args.checkpoint_dir)[0], exist_ok=True)
    train()
