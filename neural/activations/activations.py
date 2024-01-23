import numpy as np
from numpy.typing import NDArray
from neural.constants import SIGMOID_CLIPPING, LEAKINESS_FACTOR


class Activation:
    @staticmethod
    def ident(z) -> NDArray[np.number]:
        return z
    @staticmethod
    def sigmoid(z) -> NDArray[np.number]:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -SIGMOID_CLIPPING, SIGMOID_CLIPPING)))
    @staticmethod
    def relu(z) -> NDArray[np.number]:
        return np.maximum(0.0, z)
    @staticmethod
    def leaky_relu(z, alpha = LEAKINESS_FACTOR) -> NDArray[np.number]:
        return np.where(z > 0, z, alpha * z)

class ActivationDerivative:
    @staticmethod
    def ident(z) -> NDArray[np.number]:
        return np.ones_like(z)
    @staticmethod
    def sigmoid(z) -> NDArray[np.number]:
        sigmoid_value = Activation.sigmoid(z)
        return sigmoid_value * (1.0 - sigmoid_value)
    @staticmethod
    def relu(z) -> NDArray[np.number]:
        return np.where(z > 0, 1, 0)
    @staticmethod
    def leaky_relu(z, alpha = LEAKINESS_FACTOR) -> NDArray[np.number]:
        return np.where(z > 0, 1, alpha)
