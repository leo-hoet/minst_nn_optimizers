import math
import numpy as np
from keras import Input, Model
from keras.src.layers import Dense
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Función para inicializar los pesos aleatorios


def _random_weights():
    return [
        [],
        [np.random.rand(784, 128), np.random.rand(128)],  # Crea una matriz rand de 748x128 y un arreglo de 128x1
        [np.random.rand(128, 64), np.random.rand(64)],  # Crea una matriz rand de 128x64 y un arreglo de 64x1
        [np.random.rand(64, 10), np.random.rand(10)]  # Crea una matriz de 64x10 y un arreglo de 10x1
    ]


def zero_weights():
    return [
        [],
        [np.zeros((784, 128)), np.zeros(128)],
        [np.zeros((128, 64)), np.zeros(64)],
        [np.zeros((64, 10)), np.zeros(10)]
    ]


class NNModel:
    def __init__(self):  # Constructor de la red (Arquitectura)
        inputs = Input(shape=(784,))  # Inicializar el tenson con una tupla de 784 elementos
        # Selecciona relu como función de activación para todas las unidades conectadas a las entradas (784 elementos iniciales)
        x = Dense(128, activation='relu')(inputs)
        # Selecciona relu como función de activación para todas las siguientes unidades conectadas (128 siguientes)
        x = Dense(64, activation='relu')(x)
        # Selecciona softmax como función de activación para las últimas conectadas (64 finales)
        outputs = Dense(10, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs, name="mnist_model")  # Salida mediante el modelo "mnist_model"

        # Compile the model (we won't use this for training, but it's required to use the model)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model  # Red
        self.randomize_weights()  # aplicar la función de abajo ↓

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

    def randomize_weights(self):  # aplicar la función de abajo ↓
        self.set_custom_weights(_random_weights())

    def set_custom_weights(self, weights):
        shape = getattr(weights, 'shape', None)  # New line
        if shape and len(shape) == 1:  # New line
            weights = self._deflatten_weights(weights)  # New line

        for layer, w in zip(self.model.layers, weights):  # Para cada capa de la red, setée los pesos
            layer.set_weights(w)  # Según Keras doc -> Sets the values of layer.weights from a list of NumPy arrays

    def predict_digit(self, image):
        image = image.reshape(1, 784)  # cambia la estructura de la imagen a 1 fila, 784 cols
        # usa la imagen reestructurada como entrada para la predicción del modelo
        prediction = self.model.predict(image)
        return np.argmax(prediction)  # devuelve el índice del mayor número de la predicción

    def get_weights_as_numpy(self):
        weights_list = []  # crea una lista con todos los pesos de la red
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            for w in layer_weights:
                weights_list.append(w.flatten())
                # print("w's = ", layer_weights)
                # print("w size = ", len(layer_weights))
        return np.concatenate(weights_list)  # retorna la lista de todos los pesos en horizontal

    def metrics(self, X_test, y_true):
        # Reshape X_test if necessary
        # if X_test.ndim == 3:
        #   X_test = X_test.reshape(-1, 784)

        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_classes)
        precision = precision_score(y_true, y_pred_classes, average='weighted')
        recall = recall_score(y_true, y_pred_classes, average='weighted')
        f1 = f1_score(y_true, y_pred_classes, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    # New function
    def fitness(self, x_test, y_test) -> float:
        metrics = self.metrics(x_test, y_test)
        return metrics['f1_score']
