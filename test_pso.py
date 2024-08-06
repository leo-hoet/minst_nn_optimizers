import math
import unittest

import numpy as np

from model import NNModel, _random_weights, zero_weights


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
