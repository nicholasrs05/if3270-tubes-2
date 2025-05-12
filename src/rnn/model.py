import numpy as np
from typing import Optional
from sklearn.metrics import f1_score
from ..ffnn.model import FFNNModel
from ..ffnn.activation import Softmax
from ..ffnn.loss import CategoricalCrossentropy
from ..ffnn.initializer import XavierInitializer


class RNNModel:
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        rnn_units: int = 128,
        dropout_rate: float = 0.2,
        bidirectional: bool = True,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self._initialize_weights()

        rnn_output_dim = rnn_units * 2 if bidirectional else rnn_units
        self.classifier = FFNNModel(
            layers=[rnn_output_dim, 3],
            activation_functions=[Softmax()],
            loss_function=CategoricalCrossentropy(),
            weight_initializer=[XavierInitializer()],
            learning_rate=0.01,
        )

    def _initialize_weights(self):
        self.embedding_weights = np.random.normal(
            0, 0.1, (self.vocab_size, self.embedding_dim)
        )

        rnn_units = self.rnn_units * 2 if self.bidirectional else self.rnn_units

        self.Wxh = np.random.normal(0, 0.1, (self.embedding_dim, rnn_units))
        self.Whh = np.random.normal(0, 0.1, (rnn_units, rnn_units))
        self.bh = np.zeros((1, rnn_units))

    def _rnn_step(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        h = np.tanh(np.dot(x, self.Wxh) + np.dot(h_prev, self.Whh) + self.bh)
        return h

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -500, 500))

    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        rnn_units = self.rnn_units * 2 if self.bidirectional else self.rnn_units
        h = np.zeros((batch_size, rnn_units))

        for t in range(seq_length):
            x_t = self.embedding_weights[x[:, t]]
            h = self._rnn_step(x_t, h)

        if self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, h.shape) / (
                1 - self.dropout_rate
            )
            h = h * mask

        probs = self.classifier.predict(h, training=True)
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.forward_propagation(x)
        return np.argmax(probs, axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        return f1_score(y, y_pred, average="macro")

    def save_weights(self, filepath: str):
        weights = {
            'embedding_weights': self.embedding_weights,
            'Wxh': self.Wxh,
            'Whh': self.Whh,
            'bh': self.bh
        }
        np.save(filepath, weights)
        self.classifier.save(filepath + "_classifier")

    def load_weights(self, filepath: str):
        weights = np.load(filepath, allow_pickle=True).item()
        self.embedding_weights = weights['embedding_weights']
        self.Wxh = weights['Wxh']
        self.Whh = weights['Whh']
        self.bh = weights['bh']
        self.classifier = FFNNModel.load(filepath + "_classifier")

    def load_dense_layer_weights(self, filepath: str):
        data = np.load(filepath)
        self.classifier.layers[0].weights = data['weights']
        self.classifier.layers[0].biases = data['biases'].reshape(1, -1)
