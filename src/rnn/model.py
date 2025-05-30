import numpy as np
from typing import Optional
from sklearn.metrics import f1_score
from .layer import EmbeddingScratch, RNNScratch, DenseScratch


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

        self.embedding_layer = EmbeddingScratch(vocab_size, embedding_dim)
        self.rnn_layer = RNNScratch(embedding_dim, rnn_units, bidirectional)
        self.dense_layer = DenseScratch(3, activation="softmax")

    def forward_propagation(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        if self.bidirectional:
            # Forward pass
            h_f = np.zeros((batch_size, self.rnn_units))
            for t in range(seq_length):
                x_t = self.embedding_layer.forward(x[:, t])
                h_f = self.rnn_layer.forward(x_t, h_f, is_forward=True)

            # Backward pass
            h_b = np.zeros((batch_size, self.rnn_units))
            for t in range(seq_length - 1, -1, -1):
                x_t = self.embedding_layer.forward(x[:, t])
                h_b = self.rnn_layer.forward(x_t, h_b, is_forward=False)

            h = np.concatenate([h_f, h_b], axis=1)
        else:
            h = np.zeros((batch_size, self.rnn_units))
            for t in range(seq_length):
                x_t = self.embedding_layer.forward(x[:, t])
                h = self.rnn_layer.forward(x_t, h)

        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, h.shape) / (
                1 - self.dropout_rate
            )
            h = h * mask

        probs = self.dense_layer.forward(h)
        return probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.forward_propagation(x, training=False)
        return np.argmax(probs, axis=1)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        return f1_score(y, y_pred, average="macro")

    def load_weights(self, filepath: str):
        weights = np.load(filepath, allow_pickle=True)

        self.embedding_layer.set_weights(weights["embedding_weights"])

        rnn_weights = {
            "Wxh": weights["rnn_weights"].item()["Wxh"],
            "Whh": weights["rnn_weights"].item()["Whh"],
            "bh": weights["rnn_weights"].item()["bh"],
        }
        self.rnn_layer.set_weights(rnn_weights)

        dense_weights = (
            weights["dense_weights"].item()["weights"],
            weights["dense_weights"].item()["biases"],
        )
        self.dense_layer.set_weights(dense_weights)

