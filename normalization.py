import numpy as np


class Normalization:
    def z_score(input):
        class ZScoreData:
            def __init__(self, mean, std):
                self._mean = mean
                self._std = std
        input = np.array(input)
        data_normalization = []
        for feature in range(np.shape(input)[1]):
            data_normalization.append(ZScoreData(np.mean(input[:, feature]), np.std(input[:, feature])))
        def z_score_(x):
            x_norm = []
            for feature, dn in zip(x, data_normalization):
                x_norm.append((feature - dn._mean) / dn._std)
            return np.array(x_norm)
        return z_score_
    def min_max(input):
        class MinMaxData:
            def __init__(self, min, max):
                self._min = min
                self._max = max
        input = np.array(input)
        data_normalization = []
        for feature in range(np.shape(input)[1]):
            data_normalization.append(MinMaxData(np.min(input[:, feature]), np.max(input[:, feature])))
        def min_max_(x):
            x_norm = []
            for feature, dn in zip(x, data_normalization):
                x_norm.append((feature - dn._min) / (dn._max - dn._min))
            return np.array(x_norm)
        return min_max_
    def no_normalization(_input):
        def no_normalization_(x):
            return np.array(x)
        return no_normalization_