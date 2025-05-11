from abc import ABC
from src.ffnn.initializer import Initializer
from src.ffnn.activation import Activation


class Layer(ABC):
    def __init__(
        self,
        input_dim: int,
        neuron_count: int,
        activation: Activation,
        weights_initializer: Initializer,
    ):
        self.input_dim: int = input_dim
        self.neuron_count: int = neuron_count
        self.activation: Activation = activation

        self.weights = weights_initializer.initialize((input_dim, neuron_count))
        self.biases = weights_initializer.initialize((neuron_count,))
        self.Z = None
        self.A = None
        self.gradients = None

    def get_weights(self):
        return self.weights, self.biases

    def set_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases
