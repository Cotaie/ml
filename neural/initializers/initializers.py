import numpy as np
from neural.constants import MEAN, VARIANCE


class Initializers:
    @staticmethod
    def zeros(nr_inputs, nr_neurons):
        return np.zeros((nr_inputs, nr_neurons))
    @staticmethod
    def ones(nr_inputs, nr_neurons):
        return np.ones((nr_inputs, nr_neurons))
    @staticmethod
    def random_normal(nr_inputs, nr_neurons, mean=MEAN, var=VARIANCE, seed=None):
        np.random.seed(seed)
        return np.random.normal(mean, var, (nr_inputs, nr_neurons))
    @staticmethod
    def xavier_normal(nr_inputs, nr_neurons, seed=None):
        np.random.seed(seed)
        return np.random.normal(0, 2 / (nr_neurons + nr_inputs), (nr_inputs, nr_neurons))
    @staticmethod
    def xavier_uniform(nr_inputs, nr_neurons, seed=None):
        np.random.seed(seed)
        mean = 0
        var = 6 / (nr_inputs + nr_neurons)
        low = mean - np.sqrt(3 * np.square(var))
        high = mean + np.sqrt(3 * np.square(var))
        return np.random.uniform(low, high, (nr_inputs, nr_neurons))
    @staticmethod
    def he(nr_inputs, nr_neurons, seed=None):
        np.random.seed(seed)
        return np.random.normal(0, 2 / nr_inputs, (nr_inputs, nr_neurons))
