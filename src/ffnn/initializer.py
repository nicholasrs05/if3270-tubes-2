from abc import ABC, abstractmethod
import numpy as np


class Initializer(ABC):
    def __init__(self, seed=None):
        self.seed = seed

    @abstractmethod
    def initialize(self, shape):
        pass


class ZeroInitializer(Initializer):
    def initialize(self, shape):
        return np.zeros(shape)


class RandomUniformInitializer(Initializer):
    def __init__(self, lower_bound=0.0, upper_bound=1.0, seed=None):
        super().__init__(seed)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def initialize(self, shape):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.uniform(self.lower_bound, self.upper_bound, size=shape)


class RandomNormalInitializer(Initializer):
    def __init__(self, mean=0.0, variance=1.0, seed=None):
        super().__init__(seed)
        self.mean = mean
        self.variance = variance

    def initialize(self, shape):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.normal(self.mean, self.variance, size=shape)


class HeInitializer(Initializer):
    def __init__(self, seed=None):
        super().__init__(seed)

    def initialize(self, shape):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])


class XavierInitializer(Initializer):
    def __init__(self, seed=None):
        super().__init__(seed)

    def initialize(self, shape):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.randn(*shape) * np.sqrt(1.0 / shape[0])