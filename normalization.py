import numpy as np


class Normalization:
    def z_score(input):
        data_normalization = np.array([ZScoreData(np.mean(column), np.std(column)) for column in np.array(input).T])
        def z_score_(x):
            x = np.array(x, dtype=np.float64)
            std = np.array([dn._std for dn in data_normalization])
            mean =  np.array([dn._mean for dn in data_normalization])
            return np.divide(x - mean, std, out=x.copy(), where=std!=0)
        return z_score_
    def min_max(input):
        data_normalization = np.array([MinMaxData(np.min(column), np.max(column)) for column in np.array(input).T])
        def min_max_(x):
            x = np.array(x, dtype=np.float64)
            min = np.array([dn._min for dn in data_normalization])
            diff =  np.array([dn._max for dn in data_normalization]) - min
            return np.divide(x - min, diff, out=x.copy(), where=diff!=0)
        return min_max_
    def no_normalization(_input):
        def no_normalization_(x):
            return x
        return no_normalization_

class ZScoreData:
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

class MinMaxData:
    def __init__(self, min, max):
        self._min = min
        self._max = max