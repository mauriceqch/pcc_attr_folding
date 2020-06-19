import tensorflow.compat.v1 as tf
import numpy as np
from .chamfer_dist import chamfer_dist


class ChamferDistTest(tf.test.TestCase):
    def test_function_with_batch(self):
        A = tf.constant([
            [[0.0, 1.0, 2.0],
             [0.0, 1.0, 2.0]],
            [[0.0, 2.0, 4.0],
             [0.0, 2.0, 2.0]],
        ])
        B = tf.constant([
            [[1.0, 2.0, 0.0],
             [1.0, 2.0, 0.0]],
            [[1.0, 2.0, 0.0],
             [1.0, 2.0, 4.0]],
        ])
        C = tf.constant([
            (1.0 + 1.0 + 4.0) + (1.0 + 1.0 + 4.0),
            ((1.0 + 0.0 + 0.0) + (1.0 + 0.0 + 4.0)) / 2 + ((1.0 + 0.0 + 4.0) + (1.0 + 0.0 + 0.0)) / 2
        ])
        self.assertAllClose(C, chamfer_dist(A, B))


if __name__ == "__main__":
    tf.test.main()
