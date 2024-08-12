from itertools import product
import json
from typing import List
from pso import PSO
import tensorflow as tf
import numpy as np
import dataclasses


@dataclasses.dataclass
class FitnessResult:
    fitness: List[float]
    w: float
    c_local: float
    c_global: float
    pop_size: int


def dataset():
    data = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data

    x_train, y_train = x_train[:1000], y_train[:1000]
    x_test, y_test = x_test[:1000], y_test[:1000]
    return (x_train, y_train), (x_test, y_test)


def write_to_file(d: FitnessResult):
    with open('fitnesses.json', 'r+') as f:
        all_data = f.read()
        if not all_data:
            all_data_parsed = {'runs': []}
        else:
            all_data_parsed = json.loads(all_data)
        all_data_parsed['runs'].append(dataclasses.asdict(d))
        f.seek(0)
        f.write(json.dumps(all_data_parsed))


def main():
    ws = [0.2, 0.6, 0.8, 1.5]
    c_locals = [1, 5, 10, 20]
    c_globals = [1, 5, 10, 20]
    pop_sizes = [10, 15, 20]
    p = product(ws, c_locals, c_globals, pop_sizes)
    data = dataset()

    for (w, c_local, c_global, pop_size) in p:
        pso = PSO(
            data=data,
            w=w,
            c_local=c_local,
            c_global=c_global,
            n_particles=pop_size,
        )
        fitnesses = pso.train(epochs=10)
        d = FitnessResult(
            fitness=fitnesses,
            w=w,
            c_local=c_local,
            c_global=c_global,
            pop_size=pop_size
        )
        write_to_file(d)

    print(list(p))


if __name__ == "__main__":
    main()
