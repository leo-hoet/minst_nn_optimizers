import math
import unittest

import numpy as np

from model import NNModel, _random_weights, zero_weights
from pso import PSO


class PsoTestCase(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_count_params(self):
        model = NNModel()
        print("LOS PARAMS DEL MODELO SON:", model.model.count_params())
        layers = model.model.get_weights()

        for layer in layers:
            sizes = [s for s in layer.shape]
            total_size = math.prod(sizes)
            print("lAYER SIZE ", total_size, sizes)

    def test_set_weights(self):
        model = NNModel()
        zeros = np.zeros(model.len_params)

        w = model.get_weights_as_numpy()
        self.assertFalse(np.array_equal(zeros, w))

        model.set_custom_weights(zeros)
        w = model.get_weights_as_numpy()
        self.assertTrue(np.array_equal(zeros, w))

    def test_set_weights_idempotent(self):
        model = NNModel()
        w1 = model.get_weights_as_numpy()
        model.set_custom_weights(w1)
        w2 = model.get_weights_as_numpy()

        self.assertTrue(np.array_equal(w1, w2))
        import tensorflow as tf
        data = tf.keras.datasets.mnist.load_data()

        p1 = PSO(data, 1, 1, 1)
        p2 = PSO(data, 1, 2, 1)

        w1 = p1._model.get_weights_as_numpy()
        p2._model.set_custom_weights(w1)
        w2 = p2._model.get_weights_as_numpy()
        self.assertTrue(np.array_equal(w1, w2))
