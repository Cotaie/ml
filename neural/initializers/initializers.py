import numpy as np
from neural.constants import MEAN, VARIANCE


class Initializers:
    @staticmethod
    def zeros(input_dim, output_dim):
        return np.zeros((output_dim, input_dim))
    @staticmethod
    def ones(input_dim, output_dim):
        return np.ones((output_dim, input_dim))
    @staticmethod
    def random_normal(input_dim, output_dim, mean=MEAN, var=VARIANCE, seed=None):
        np.random.seed(seed)
        return np.random.normal(mean, var, (output_dim, input_dim))
    @staticmethod
    def xavier_normal(input_dim, output_dim, seed=None):
        np.random.seed(seed)
        return np.random.normal(0, 2 / (output_dim + input_dim), (output_dim, input_dim))
    @staticmethod
    def xavier_uniform(input_dim, output_dim, seed=None):
        np.random.seed(seed)
        mean = 0
        var = 6 / (input_dim + output_dim)
        low = mean - np.sqrt(3 * np.square(var))
        high = mean + np.sqrt(3 * np.square(var))
        return np.random.uniform(low, high, (output_dim, input_dim))
    @staticmethod
    def he(input_dim, output_dim, seed=None):
        np.random.seed(seed)
        return np.random.normal(0, 2 / input_dim, (output_dim, input_dim))
