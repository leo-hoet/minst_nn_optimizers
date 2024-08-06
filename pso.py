
from model import NNModel


class PSO:
    def __init__(self, data, w: float, c_local: float, c_global: float, n_particles=100, model: NNModel | None = None):
        (x_train, y_train), (x_test, y_test) = data
        x_train, x_test = x_train / 255.0, x_test / 255.0

        self._x_train = x_train.reshape(x_train.shape[0], 784)
        self._y_train = y_train
        self._x_test = x_test.reshape(x_test.shape[0], 784)
        self._y_test = y_test

        self._w = w
        self._c1 = c_local
        self._c2 = c_global

        self._model = NNModel()
        self._particles = [Particle() for _ in range(n_particles)]

    def run(self, epochs: int = 100) -> Best:
        raise NotImplementedError("TODO ESTO")
