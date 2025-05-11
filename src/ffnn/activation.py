from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dout):
        pass


class Linear(Activation):
    def forward(self, x):
        return x

    def backward(self, dout):
        return 1


class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, output):
        return np.clip(output, 1e-10, 1 - 1e-10) * (
            1 - np.clip(output, 1e-10, 1 - 1e-10)
        )


class ReLU(Activation):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * (self.x > 0)


class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, output):
        return 1 - output**2


class Softmax(Activation):
    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def backward(self, output):
        batch_size, n_class = output.shape
        jacobian = np.zeros((batch_size, n_class, n_class))

        for b in range(batch_size):
            o = output[b].reshape(-1, 1)
            jacobian[b] = np.diagflat(output[b]) - np.dot(o, o.T)

        return jacobian


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, dout):
        return dout * np.where(self.x > 0, 1, self.alpha)


class ELU(Activation):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        self.out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        return self.out

    def backward(self, dout):
        return dout * np.where(self.x > 0, 1, self.out + self.alpha)
