import numpy as np
from neural.constants import SIGMOID_CLIPPING, LEAKINESS_FACTOR


class Activation:
    @staticmethod
    def ident(z: float):
        return z
    @staticmethod
    def sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -SIGMOID_CLIPPING, SIGMOID_CLIPPING)))
    @staticmethod
    def relu(z: float):
        return np.maximum(0.0, z)
    @staticmethod
    def leaky_relu(z: float, alpha = LEAKINESS_FACTOR):
        return np.where(z > 0, z, alpha * z)

class ActivationDerivative:
    @staticmethod
    def ident(_z: float):
        return 1
    @staticmethod
    def sigmoid(z: float):
        sigmoid_value = Activation.sigmoid(z)
        return sigmoid_value * (1.0 - sigmoid_value)
    @staticmethod
    def relu(z: float):
        if isinstance(z, list):
            z = np.array(z)
        return np.where(z > 0, 1, 0)
    @staticmethod
    def leaky_relu(z: float, alpha = LEAKINESS_FACTOR):
        return np.where(z > 0, 1, alpha)
