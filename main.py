import time

import tensorflow as tf

from model import NNModel, zero_weights
from pso import PSO


def some_tests():
    # Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize pixels between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Flatten the images
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    model = NNModel()

    def run():
        t0 = time.perf_counter()
        model.randomize_weights()
        metrics = model.metrics(x_test, y_test)
        t1 = time.perf_counter()
        print(f"ELAPSED TIME {t1-t0} ms {metrics=}")

    model.set_custom_weights(zero_weights())
    metrics = model.metrics(x_test, y_test)
    print(metrics)

    model.randomize_weights()
    metrics = model.metrics(x_test, y_test)
    print(metrics)


def main():
    data = tf.keras.datasets.mnist.load_data()
    p = PSO(
        data=data,
        w=0.2,
        c_local=10,
        c_global=10,
        n_particles=30
    )
    p.run(epochs=20)


if __name__ == '__main__':
    main()
