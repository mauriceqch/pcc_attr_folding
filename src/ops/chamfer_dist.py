import tensorflow.compat.v1 as tf
from .nn_distance import nn_distance, nn_distance2


FUNCTIONS = {
    0: nn_distance,
    1: nn_distance2
}


def expand_if_needed(x):
    if len(x.shape) == 2:
        x = tf.expand_dims(x, axis=0)
    return x


def chamfer_dist(A, B, k=0, data_format='channels_last'):
    """
    Computes the chamfer distance between A and B.

    Args:
        A: ? x n x 3 tensor
        B: ? x m x 3 tensor

    Returns:
        ? tensor
    """
    # ? x n x m
    assert data_format == 'channels_last'
    assert A.shape[-1] == 3
    assert B.shape[-1] == 3
    assert k in FUNCTIONS.keys()
    with tf.variable_scope('chamfer_dist'):
        A = expand_if_needed(A)
        B = expand_if_needed(B)
        # ? x n, ? x m
        AB_dists, idx_fwd, BA_dists, idx_bwd = FUNCTIONS[k](A, B)
        # ?
        AB_mdist = tf.reduce_mean(AB_dists, axis=-1)
        # ?
        BA_mdist = tf.reduce_mean(BA_dists, axis=-1)

        return tuple(tf.squeeze(x, axis=[0]) for x in [AB_mdist + BA_mdist, AB_dists, idx_fwd, BA_dists, idx_bwd])

