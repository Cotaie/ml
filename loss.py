import numpy as np
from constants import EPSILON


class Loss:
    def quadratic(g: float, y_i: float) -> float:
        return np.square(g - y_i)
    def absolute(g: float, y_i: float):
        return np.abs(g - y_i)
    def log(g: float, y_i: float, eps=EPSILON) -> float:
        return -(y_i * np.log(g + eps) + (1 - y_i) * np.log(1 - g + eps))

class LossDerivative:
    def quadratic(g, y_i):
        return 2 * (g - y_i)
    def absolute(g, y_i):
        return np.where(g > y_i, 1, np.where(g < y_i, -1, 0))
    def log(g, y_i, eps=EPSILON):
        return -y_i / (g + eps) + (1 - y_i) / (1 - g + eps)

