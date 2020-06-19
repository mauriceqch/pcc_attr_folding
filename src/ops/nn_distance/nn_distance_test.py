#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
import numpy as np
import tensorflow.compat.v1 as tf
from nn_distance import nn_distance
from nn_distance2 import nn_distance2


def simple_nn(xyz1, xyz2):
    def is_valid_shape(shape):
        return len(shape) == 3 and shape[-1] == 3
    assert(is_valid_shape(xyz1.shape))
    assert(is_valid_shape(xyz2.shape))
    assert(xyz1.shape[0] == xyz2.shape[0])
    diff = np.expand_dims(xyz1, -2) - np.expand_dims(xyz2, -3)
    square_dst = np.sum(diff**2, axis=-1)
    dst1 = np.min(square_dst, axis=-1)
    dst2 = np.min(square_dst, axis=-2)
    idx1 = np.argmin(square_dst, axis=-1)
    idx2 = np.argmin(square_dst, axis=-2)
    return dst1, idx1, dst2, idx2


def simple_nn_k(xyz1, xyz2, k):
    def is_valid_shape(shape):
        return len(shape) == 3 and shape[-1] == 3
    assert(is_valid_shape(xyz1.shape))
    assert(is_valid_shape(xyz2.shape))
    assert(xyz1.shape[0] == xyz2.shape[0])
    diff = np.expand_dims(xyz1, -2) - np.expand_dims(xyz2, -3)
    square_dst = np.sum(diff**2, axis=-1)
    idx1 = np.argpartition(square_dst, k, axis=-1)[:, :, k]
    idx2 = np.argpartition(square_dst, k, axis=-2)[:, k, :]
    dst1 = np.zeros(idx1.shape)
    dst2 = np.zeros(idx2.shape)
    for i in range(idx1.shape[0]):
        for j in range(idx1.shape[1]):
            dst1[i, j] = square_dst[i, j, idx1[i, j]]
    for i in range(idx2.shape[0]):
        for j in range(idx2.shape[1]):
            dst2[i, j] = square_dst[i, idx2[i, j], j]
    return dst1, idx1, dst2, idx2


def tf_nn(xyz1, xyz2, device, dist_function):
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(device):
            xyz1 = tf.constant(xyz1)
            xyz2 = tf.constant(xyz2)
            nn = dist_function(xyz1, xyz2)

    with tf.Session(graph=graph) as sess:
        actual = sess.run(nn)
    return actual

devices = ['/cpu:0', '/gpu:0']


class TestNnDistance(unittest.TestCase):

    def _compare_values(self, actual, expected):

        self.assertEqual(len(actual), len(expected))
        # distances
        for i in [0, 2]:
            # TF 1.15, g++ 7.4, cuda 10.0, cudnn 7.4.2, GTX 1080 Ti
            # relative difference slightly exceeds 1e-7
            np.testing.assert_allclose(actual[i], expected[i], rtol=2e-7)
        # indices
        for i in [1, 3]:
            np.testing.assert_equal(actual[i], expected[i])

    def _compare(self, xyz1, xyz2, expected):
        for device in devices:
            actual = tf_nn(xyz1, xyz2, device, nn_distance)
            self._compare_values(actual, expected)

    def _compare2(self, xyz1, xyz2, expected):
        for device in devices:
            actual = tf_nn(xyz1, xyz2, device, nn_distance2)
            self._compare_values(actual, expected)

    def test_small(self):
        xyz1 = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        xyz2 = np.array([[[-100, 0, 0], [2, 0, 0]]], dtype=np.float32)
        expected = \
            np.array([[4, 1, 5]]), \
            np.array([[1, 1, 1]]), \
            np.array([[10000, 1]]), \
            np.array([[0, 1]])
        self._compare(xyz1, xyz2, expected)

    def test_small_nn2(self):
        xyz1 = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        xyz2 = np.array([[[-100, 0, 0], [2, 0, 0]]], dtype=np.float32)
        expected = \
            np.array([[10000, 10201, 10001]]), \
            np.array([[0, 0, 0]]), \
            np.array([[10001, 4]]), \
            np.array([[2, 0]])
        self._compare2(xyz1, xyz2, expected)

    def test_big(self):
        batch_size = 5
        n1 = 10
        n2 = 20
        xyz1 = np.random.randn(batch_size, n1, 3).astype(np.float32)
        xyz2 = np.random.randn(batch_size, n2, 3).astype(np.float32)
        expected = simple_nn(xyz1, xyz2)
        self._compare(xyz1, xyz2, expected)

    def test_big_nn2(self):
        batch_size = 5
        n1 = 10
        n2 = 20
        xyz1 = np.random.randn(batch_size, n1, 3).astype(np.float32)
        xyz2 = np.random.randn(batch_size, n2, 3).astype(np.float32)
        expected = simple_nn_k(xyz1, xyz2, 1)
        self._compare2(xyz1, xyz2, expected)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    unittest.main()
