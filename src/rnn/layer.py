import numpy as np
from abc import ABC, abstractmethod


class LayerScratch(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, input):
        pass


class EmbeddingScratch(LayerScratch):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = None

    def set_weights(self, weights: np.ndarray):
        if weights.shape != (self.vocab_size, self.embedding_dim):
            raise ValueError(
                f"Expected weights shape ({self.vocab_size}, {self.embedding_dim}), got {weights.shape}"
            )
        self.weights = weights

    def forward(self, input: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Weights not set. Call set_weights first.")
        return self.weights[input]


class RNNScratch(LayerScratch):
    def __init__(self, input_dim: int, hidden_dim: int, bidirectional: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None

    def set_weights(self, weights: dict):
        if not all(k in weights for k in ["Wxh", "Whh", "bh"]):
            raise ValueError("Missing required weights: Wxh, Whh, bh")

        self.kernel = weights["Wxh"]
        self.recurrent_kernel = weights["Whh"]
        self.bias = weights["bh"]

    def forward(
        self, x: np.ndarray, h_prev: np.ndarray, is_forward: bool = True
    ) -> np.ndarray:
        if any(w is None for w in [self.kernel, self.recurrent_kernel, self.bias]):
            raise ValueError("Weights not set. Call set_weights first.")

        if self.bidirectional:
            half_units = self.hidden_dim
            if is_forward:
                # Forward pass
                h = np.tanh(
                    np.dot(x, self.kernel[:, :half_units])
                    + np.dot(h_prev, self.recurrent_kernel[:half_units, :half_units])
                    + self.bias[:half_units]
                )
            else:
                # Backward pass
                h = np.tanh(
                    np.dot(x, self.kernel[:, half_units:])
                    + np.dot(h_prev, self.recurrent_kernel[half_units:, half_units:])
                    + self.bias[half_units:]
                )
        else:
            h = np.tanh(
                np.dot(x, self.kernel)
                + np.dot(h_prev, self.recurrent_kernel)
                + self.bias
            )

        return h


class DenseScratch(LayerScratch):
    def __init__(self, neurons, activation=None):
        super().__init__()
        self.neurons = neurons
        self.activation = activation
        self.weights = None
        self.bias = None

    def set_weights(self, weights):
        w, b = weights

        if len(w.shape) != 2:
            raise ValueError(f"Expected weights to be 2D, got shape {w.shape}")
        if w.shape[1] != self.neurons:
            raise ValueError(
                f"Weights don't match neurons: expected {self.neurons}, got {w.shape[1]}"
            )
        if b.shape != (self.neurons,):
            raise ValueError(
                f"Bias shape mismatch: expected ({self.neurons},), got {b.shape}"
            )

        self.weights = w
        self.bias = b

    def forward(self, input):
        if self.weights.shape[0] != input.shape[1]:
            raise ValueError(
                f"Weights shape {self.weights.shape[0]} doesn't match input shape {input.shape[1]}"
            )

        input = np.clip(input, -1e6, 1e6)
        input = input / (np.max(np.abs(input)) + 1e-8)

        self.weights = np.clip(self.weights, -1e6, 1e6)
        self.weights = self.weights / (np.max(np.abs(self.weights)) + 1e-8)

        self.bias = np.clip(self.bias, -1e6, 1e6)
        self.bias = self.bias / (np.max(np.abs(self.bias)) + 1e-8)

        output = np.dot(input, self.weights) + self.bias

        if self.activation == "relu":
            output = np.maximum(0, output)
        elif self.activation == "sigmoid":
            output = np.clip(output, -500, 500)
            output = 1 / (1 + np.exp(-output))
        elif self.activation == "softmax":
            output = np.clip(output, -500, 500)
            shifted_output = output - np.max(output, axis=-1, keepdims=True)
            exp_output = np.exp(shifted_output)
            output = exp_output / (np.sum(exp_output, axis=-1, keepdims=True) + 1e-8)

        return output

