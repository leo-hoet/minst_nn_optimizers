
import collections
import numpy as np
from sklearn.metrics import classification_report
from model import NNModel


class PSO:
    def __init__(self, data, w: float, c_local: float, c_global: float, n_particles=100, model: NNModel | None = None):
        (x_train, y_train), (x_test, y_test) = data
        x_train, x_test = x_train / 255.0, x_test / 255.0

        self._x_train = x_train.reshape(x_train.shape[0], 784)
        self._y_train = y_train
        self._x_test = x_test.reshape(x_test.shape[0], 784)[:100]
        self._y_test = y_test[:100]
        print(collections.Counter(self._y_test))  # Print ocurrences of each class

        self._w = w
        self._c1 = c_local
        self._c2 = c_global

        self._n_particles = n_particles
        self._model = NNModel()

    def randomize_arr(self):
        return np.random.random((self._n_particles, self._model.len_params))

    def ind_fitness(self, pos) -> float:
        self._model.set_custom_weights(pos)
        return self._model.fitness(self._x_test, self._y_test)

    def update_vel(self, r1, r2, curr_vel, bests_pos, pos, swarm_best):
        new_vel = self._w * curr_vel  # Interia
        new_vel += (self._c1 * r1 * (bests_pos - pos).T).T  # Local best
        new_vel += (self._c2 * r2 * (swarm_best - pos).T).T  # Global best

        assert new_vel.shape == curr_vel.shape, "New vel has not the same shape as input"

        return new_vel

    def print_report(self, w):
        y_true = self._y_test
        self._model.set_custom_weights(w)

        predicted = self._model.model.predict(self._x_test)
        predicted = np.argmax(predicted, axis=1)

        print([(x,  z) for (x,  z) in zip(y_true,  predicted)])
        res = classification_report(y_true=y_true, y_pred=predicted)
        print(res)

    def run(self, epochs: int = 100):
        def f(row):
            return self.ind_fitness(row)

        pos = self.randomize_arr()
        bests = np.copy(pos)
        vel = self.randomize_arr()
        fitness = np.apply_along_axis(f, 1, pos)

        swarm_best_pos_idx = np.argmax(fitness)
        swarm_best_pos = pos[swarm_best_pos_idx]
        swarn_best_fitness = np.max(fitness)

        for i in range(epochs):

            r1 = np.random.uniform(0, 1, size=self._n_particles)
            r2 = np.random.uniform(0, 1, size=self._n_particles)

            vel = self.update_vel(r1, r2, vel, bests, pos, swarm_best_pos)

            assert vel.shape == pos.shape
            pos += vel

            # Update bests positions
            new_fitness = np.apply_along_axis(f, 1, pos)
            idx_better_fitness = np.where(new_fitness > fitness)
            bests[idx_better_fitness] = pos[idx_better_fitness]

            # Update global best
            if np.max(new_fitness) > swarn_best_fitness:
                swarm_best_pos_idx = np.argmax(new_fitness)
                swarn_best_fitness = new_fitness[swarm_best_pos_idx]
                swarm_best_pos = pos[swarm_best_pos_idx]

            print(f"Iter {i} best fitness: {np.max(new_fitness)}")
        self.print_report(swarm_best_pos)
