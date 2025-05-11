from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def compute(self, y, t):
        pass

    @abstractmethod
    def backward(self, y, t):
        pass


class MeanSquaredError(LossFunction):
    def compute(self, y, t):
        return np.mean((y - t) ** 2)

    def backward(self, y, t):
        return -2 * (t - y)


class BinaryCrossentropy(LossFunction):
    def compute(self, y, t):
        epsilon = 1e-9
        y = np.clip(y, epsilon, 1 - epsilon)
        return -np.mean(t * np.log(y) + (1 - t) * np.log(1 - y))

    def backward(self, y, t):
        epsilon = 1e-9
        y = np.clip(y, epsilon, 1 - epsilon)
        return -1 * (t / y - (1 - t) / (1 - y))


class CategoricalCrossentropy(LossFunction):
    def compute(self, y, t):
        epsilon = 1e-9
        y = np.clip(y, epsilon, 1 - epsilon)

        if t.ndim == 1 or t.shape[1] != y.shape[1]:
            t = np.eye(y.shape[1])[t.astype(int)]

        return -np.sum(t * np.log(y)) / y.shape[0]

    def backward(self, y, t):
        return (y - t) / y.shape[0]
