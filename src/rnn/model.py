import numpy as np
from typing import Optional
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
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
        embedding_matrix: Optional[np.ndarray] = None,
        rnn_units: int = 128,
        dropout_rate: float = 0.2,
        bidirectional: bool = True,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        self.model = self._build_keras_model(embedding_matrix)
        self._initialize_weights()

        rnn_output_dim = rnn_units * 2 if bidirectional else rnn_units
        self.classifier = FFNNModel(
            layers=[rnn_output_dim, 3],
            activation_functions=[Softmax()],
            loss_function=CategoricalCrossentropy(),
            weight_initializer=[XavierInitializer()],
            learning_rate=0.01,
        )

    def _build_keras_model(self, embedding_matrix: Optional[np.ndarray]) -> Model:
        inputs = layers.Input(shape=(None,))

        if embedding_matrix is not None:
            embedding = layers.Embedding(
                self.vocab_size,
                self.embedding_dim,
                weights=[embedding_matrix],
                trainable=False,
            )(inputs)
        else:
            embedding = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)

        if self.bidirectional:
            rnn = layers.Bidirectional(
                layers.SimpleRNN(self.rnn_units, return_sequences=False)
            )(embedding)
        else:
            rnn = layers.SimpleRNN(self.rnn_units, return_sequences=False)(embedding)

        dropout = layers.Dropout(self.dropout_rate)(rnn)
        outputs = layers.Dense(3, activation="softmax")(dropout)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
        )

        return model

    def _initialize_weights(self):
        self.embedding_weights = np.random.normal(
            0, 0.1, (self.vocab_size, self.embedding_dim)
        )

        # For bidirectional RNN, we need double the units
        rnn_units = self.rnn_units * 2 if self.bidirectional else self.rnn_units

        self.Wxh = np.random.normal(0, 0.1, (self.embedding_dim, rnn_units))
        self.Whh = np.random.normal(0, 0.1, (rnn_units, rnn_units))
        self.bh = np.zeros((1, rnn_units))

        self.Why = np.random.normal(0, 0.1, (rnn_units, 3))
        self.by = np.zeros((1, 3))

    def _rnn_step(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        h = np.tanh(np.dot(x, self.Wxh) + np.dot(h_prev, self.Whh) + self.bh)
        return h

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -500, 500))

    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        # Initialize hidden state with correct dimensions
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

        logits = np.dot(h, self.Why) + self.by
        probs = self._softmax(logits)

        return probs

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def predict(self, x: np.ndarray, from_scratch: bool = False) -> np.ndarray:
        if from_scratch:
            probs = self.forward_propagation(x)
            return np.argmax(probs, axis=1)
        else:
            return np.argmax(self.model.predict(x), axis=1)

    def evaluate(
        self, x: np.ndarray, y: np.ndarray, from_scratch: bool = False
    ) -> float:
        y_pred = self.predict(x, from_scratch)
        return f1_score(y, y_pred, average="macro")

    def save_weights(self, filepath: str):
        self.model.save_weights(filepath)
        self.classifier.save(filepath + "_classifier")

    def load_weights(self, filepath: str):
        self.model.load_weights(filepath)
        self.classifier = FFNNModel.load(filepath + "_classifier")
        self._update_scratch_weights()

    def _update_scratch_weights(self):
        keras_weights = self.model.get_weights()

        self.embedding_weights = keras_weights[0]

        if self.bidirectional:
            Wxh_forward = keras_weights[1]
            Whh_forward = keras_weights[2]
            bh_forward = keras_weights[3]

            Wxh_backward = keras_weights[4]
            Whh_backward = keras_weights[5]
            bh_backward = keras_weights[6]

            self.Wxh = np.concatenate([Wxh_forward, Wxh_backward], axis=1)
            self.Whh = np.block(
                [
                    [Whh_forward, np.zeros((self.rnn_units, self.rnn_units))],
                    [np.zeros((self.rnn_units, self.rnn_units)), Whh_backward],
                ]
            )

            self.bh = np.concatenate([bh_forward, bh_backward])
            self.bh = self.bh.reshape(1, -1)

            self.Why = keras_weights[7]
            self.by = keras_weights[8]
        else:
            self.Wxh = keras_weights[1]
            self.Whh = keras_weights[2]
            self.bh = keras_weights[3].reshape(1, -1)
            self.Why = keras_weights[4]
            self.by = keras_weights[5]
