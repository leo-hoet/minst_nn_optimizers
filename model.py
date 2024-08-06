import math
import numpy as np
from keras import Input, Model
from keras.src.layers import Dense
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def _random_weights():
    return [
        [],
        [np.random.rand(784, 128), np.random.rand(128)],
        [np.random.rand(128, 64), np.random.rand(64)],
        [np.random.rand(64, 10), np.random.rand(10)]
    ]


def zero_weights():
    return [
        [],
        [np.zeros((784, 128)), np.zeros(128)],
        [np.zeros((128, 64)), np.zeros(64)],
        [np.zeros((64, 10)), np.zeros(10)]
    ]


class NNModel:
    def __init__(self):
        inputs = Input(shape=(784,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(10, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs, name="mnist_model")

        # Compile the model (we won't use this for training, but it's required to use the model)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model
        self.randomize_weights()

    @property
    def len_params(self):
        return self.model.count_params()

    def _deflatten_weights(self, w):
        weights = []
        start = 0

        # Empty list for the input layer
        weights.append([])

        # First hidden layer (784, 128) and bias (128,)
        end = start + 784 * 128
        weights.append([w[start:end].reshape(784, 128)])
        start = end

        end = start + 128
        weights[-1].append(w[start:end])
        start = end

        # Second hidden layer (128, 64) and bias (64,)
        end = start + 128 * 64
        weights.append([w[start:end].reshape(128, 64)])
        start = end

        end = start + 64
        weights[-1].append(w[start:end])
        start = end

        # Output layer (64, 10) and bias (10,)
        end = start + 64 * 10
        weights.append([w[start:end].reshape(64, 10)])
        start = end

        end = start + 10
        weights[-1].append(w[start:end])

        return weights

    def randomize_weights(self):
        self.set_custom_weights(_random_weights())

    def set_custom_weights(self, weights):
        shape = getattr(weights, 'shape', None)
        if shape and len(shape) == 1:
            weights = self._deflatten_weights(weights)

        for layer, w in zip(self.model.layers, weights):
            layer.set_weights(w)

    def predict_digit(self, image):
        prediction = self.model.predict(image)
        return np.argmax(prediction)

    def get_weights_as_numpy(self):
        weights_list = []
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            for w in layer_weights:
                weights_list.append(w.flatten())
        return np.concatenate(weights_list)

    def metrics(self, X_test, y_true):
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_classes)
        # precision = precision_score(y_true, y_pred_classes, average='weighted')
        recall = recall_score(y_true, y_pred_classes, average='weighted')
        f1 = f1_score(y_true, y_pred_classes, average='weighted')

        return {
            'accuracy': accuracy,
            # 'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def fitness(self, x_test, y_test) -> float:
        metrics = self.metrics(x_test, y_test)
        return metrics['accuracy']
