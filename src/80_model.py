import tensorflow.compat.v1 as tf
from ops.chamfer_dist import chamfer_dist
from utils import color_space

GEO_DIM = 3
data_format = 'channels_last'
channels_axis = 1 if data_format == 'channels_last' else 0
points_axis = 0 if data_format == 'channels_last' else 1


# Specialized Conv1D for stride 1 and no padding, equivalent of MLP for each sample
class Conv1D:
    def __init__(self, input_dim, output_dim, activation=tf.nn.relu, use_bias=True, name='conv1d', data_format='channels_last'):
        assert data_format in ['channels_last', 'channels_first']
        W_shape = (input_dim, output_dim) if data_format == 'channels_last' else (output_dim, input_dim)
        b_shape = (1, output_dim) if data_format == 'channels_last' else (output_dim, 1)
        self.data_format = data_format
        self.use_bias = use_bias
        self.name = name
        with tf.variable_scope(self.name) as vs:
            self.W = tf.get_variable('W', W_shape)
            if use_bias:
                self.b = tf.get_variable('b', b_shape)
            self.activation = activation
        self.vs = vs

    def __call__(self, x):
        with tf.variable_scope(self.vs.original_name_scope):
            if self.data_format == 'channels_first':
                x = self.W @ x
            else:
                x = x @ self.W
            if self.use_bias:
                x = x + self.b
            if self.activation is not None:
                x = self.activation(x)
            return x


# Sequence of 1D convolutions
class Conv1DSequence:
    def __init__(self, feature_dims, activation=tf.nn.relu, use_bias=True, name='conv1d_sequence', data_format=data_format):
        self.name = name
        with tf.variable_scope(self.name) as vs:
            self.convs = []
            for i in range(len(feature_dims) - 1):
                self.convs.append(Conv1D(feature_dims[i], feature_dims[i+1], name=f'conv1d_{i}',
                                         activation=activation, use_bias=use_bias, data_format=data_format))
        self.vs = vs

    def __call__(self, x):
        with tf.variable_scope(self.vs.original_name_scope):
            for c in self.convs:
                x = c(x)
            return x


class Encoder:
    def __init__(self, feature_dims, name='encoder'):
        self.name = name
        with tf.variable_scope(self.name) as vs:
            self.convs = Conv1DSequence(feature_dims, activation=tf.nn.relu, use_bias=True, data_format=data_format)
        self.vs = vs

    def __call__(self, x):
        with tf.variable_scope(self.vs.original_name_scope):
            y = self.convs(x)
            y = tf.reduce_max(y, axis=points_axis)
            y = tf.expand_dims(y, points_axis)
            return y


class FoldingLayer:
    def __init__(self, input_dim, filters, final_dim, use_bias=False, name='folding'):
        self.name = name
        with tf.variable_scope(self.name) as vs:
            self.convs = Conv1DSequence((input_dim, filters, final_dim),
                                        activation=tf.nn.leaky_relu, use_bias=use_bias, data_format=data_format)
        self.vs = vs

    def __call__(self, x, y):
        with tf.variable_scope(self.vs.original_name_scope):
            x = tf.concat([x, y], channels_axis)
            x = self.convs(x)
            return x


class Decoder:
    def __init__(self, input_dim, filters, y_dim, final_dim, name='decoder'):
        self.name = name
        with tf.variable_scope(self.name) as vs:
            self.layers = [
                FoldingLayer(input_dim + y_dim, filters, filters, name=f'folding_1'),
                FoldingLayer(filters + y_dim, filters, final_dim, name=f'folding_2'),
            ]
        self.n_foldings = len(self.layers)
        self.vs = vs

    def __call__(self, y, grid):
        with tf.variable_scope(self.vs.original_name_scope):
            tile_multiples = [1, 1]
            tile_multiples[points_axis] = tf.shape(grid)[points_axis]
            self.y = tf.tile(y, tile_multiples)

            self.xs = [grid]
            for i in range(self.n_foldings):
                self.xs.append(self.layers[i](self.xs[i], self.y))
            self.x_tilde = self.xs[-1]

            return self.x_tilde


class Model:
    def __init__(self, input_pipeline, filter_sizes=(128, 128, 128, 128), filters=64, ndim=3):
        points = input_pipeline.next_element
        assert points.shape[0] == 1
        self.x = points[0, :, :GEO_DIM]
        self.ori_colors = points[0, :, GEO_DIM:]

        self.encoder = Encoder((GEO_DIM,) + filter_sizes)
        self.decoder = Decoder(ndim, filters, filter_sizes[-1], ndim)

        self.y = self.encoder(self.x)
        self.grid = tf.placeholder(tf.float32, shape=(None, ndim))
        self.x_tilde = self.decoder(self.y, self.grid)

        # Losses
        cdist, AB_dists, idx_fwd, BA_dists, _ = chamfer_dist(self.x, self.x_tilde, k=0)
        _, AB_dists2, _, _, _ = chamfer_dist(self.x_tilde, self.x_tilde, k=1)
        num_points = tf.shape(self.x_tilde)[points_axis]
        # Note: no gradients flow through color_mapping, this is for information purposes
        with tf.variable_scope('color_mapping'):
            # Map colors forward
            self.colors_tilde = tf.math.unsorted_segment_mean(self.ori_colors, idx_fwd, num_points)
            # Map colors backward
            self.ori_colors_tilde = tf.gather(self.colors_tilde, idx_fwd, axis=points_axis, batch_dims=0)

            bt709_rgb_to_yuv_m = tf.constant(color_space.bt709_rgb_to_yuv_m, dtype=tf.float32)
            self.ori_colors_yuv = self.ori_colors @ bt709_rgb_to_yuv_m
            self.ori_colors_tilde_yuv = self.ori_colors_tilde @ bt709_rgb_to_yuv_m

            self.loss_weights = tf.constant([0.8, 0.1, 0.1], dtype=tf.float32)
            self.col_losses = tf.reduce_mean(
                tf.squared_difference(self.ori_colors_yuv, self.ori_colors_tilde_yuv), axis=points_axis)
            self.col_loss = tf.tensordot(self.col_losses, self.loss_weights, 1)

        self.rec_loss = tf.reduce_mean(cdist)
        # self.rec_loss = tf.constant(0.0)
        self.rep_loss = tf.math.reduce_variance(AB_dists2)
        # self.rep_loss = tf.constant(0.0)
        self.loss = self.rec_loss + self.rep_loss

        # Optimizer
        self.step = tf.train.get_or_create_global_step()
        self.lr = 1e-3
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.step)

        # Summaries
        tf.summary.scalar("rec_loss", self.rec_loss)
        tf.summary.scalar("rep_loss", self.rep_loss)
        tf.summary.scalar("col_loss", self.col_loss)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("lr", self.lr)
        tf.summary.histogram("y", self.y)
        for i in range(self.x.shape[channels_axis]):
            tf.summary.histogram(f"dim_{i}/x", self.x[:, i])
            tf.summary.histogram(f"dim_{i}/x_tilde", self.x_tilde[:, i])
