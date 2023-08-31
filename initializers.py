import numpy as np
from typing import Callable
from constants import MEAN, VARIANCE

class Initializers:
    def zeros(input_dim, output_dim):
        return np.hstack((np.zeros((output_dim, 1)),np.zeros((output_dim, input_dim))))
    def ones(input_dim, output_dim):
        return np.hstack((np.zeros((output_dim, 1)),np.ones((output_dim, input_dim))))
    def random_normal(input_dim, output_dim, mean=MEAN, var=VARIANCE):
        return np.hstack((np.zeros((output_dim, 1)),np.random.normal(mean, var, (output_dim, input_dim))))
    def xavier_normal(input_dim, output_dim):
        return np.hstack((np.zeros((output_dim, 1)),np.random.normal(0, 2/(output_dim + input_dim), (output_dim, input_dim))))
    def xavier_uniform(input_dim, output_dim):
        mean = 0
        var = 6/(input_dim + output_dim)
        low = mean - np.sqrt(3 * np.square(var))
        high = mean + np.sqrt(3 * np.square(var))
        return np.hstack((np.zeros((output_dim, 1)),np.random.uniform(low, high, (output_dim, input_dim))))
    def he(input_dim, output_dim):
        return np.hstack((np.zeros((output_dim, 1)),np.random.normal(0, 2/input_dim, (output_dim, input_dim))))


map_activations_initializers: dict[str, Callable] = {
    'linear': Initializers.random_normal,
    'sigmoid': Initializers.xavier_normal,
    'relu': Initializers.he,
    'leaky_relu': Initializers.he
}