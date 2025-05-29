import numpy as np
from typing import Optional
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import f1_score


class RNNKerasModel:
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

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.model.predict(x), axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        return f1_score(y, y_pred, average="macro")

    def save_weights(self, filepath: str):
        embedding_weights = self.model.layers[1].get_weights()[0]

        rnn_layer = self.model.layers[2]
        if self.bidirectional:
            forward_layer = rnn_layer.forward_layer
            backward_layer = rnn_layer.backward_layer

            Wxh_f = forward_layer.get_weights()[0]
            Whh_f = forward_layer.get_weights()[1]
            bh_f = forward_layer.get_weights()[2]

            Wxh_b = backward_layer.get_weights()[0]
            Whh_b = backward_layer.get_weights()[1]
            bh_b = backward_layer.get_weights()[2]

            Wxh = np.concatenate([Wxh_f, Wxh_b], axis=1)
            Whh = np.block(
                [
                    [Whh_f, np.zeros((self.rnn_units, self.rnn_units))],
                    [np.zeros((self.rnn_units, self.rnn_units)), Whh_b],
                ]
            )

            bh = np.concatenate([bh_f, bh_b])  # (2*rnn_units,)
        else:
            Wxh = rnn_layer.get_weights()[0]  # kernel
            Whh = rnn_layer.get_weights()[1]  # recurrent_kernel
            bh = rnn_layer.get_weights()[2]  # bias

        dense_weights = self.model.layers[-1].get_weights()

        weights = {
            "embedding_weights": embedding_weights,
            "rnn_weights": {"Wxh": Wxh, "Whh": Whh, "bh": bh},
            "dense_weights": {"weights": dense_weights[0], "biases": dense_weights[1]},
        }

        np.savez(filepath, **weights)

