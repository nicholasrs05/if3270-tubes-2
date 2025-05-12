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
        self.model.save_weights(filepath)

    def load_weights(self, filepath: str):
        self.model.load_weights(filepath)

    def save_dense_layer_weights(self, filepath: str):
        dense_layer = self.model.layers[-1]
        weights = dense_layer.get_weights()
        np.savez(filepath, weights=weights[0], biases=weights[1]) 