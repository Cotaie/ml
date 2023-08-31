import numpy as np
from constants import SIGMOID_CLIPPING, LEAKINESS_FACTOR


class Activation:
    def ident(z: float):
        return z
    def sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -SIGMOID_CLIPPING, SIGMOID_CLIPPING)))
    def relu(z: float):
        return np.maximum(0.0, z)
    def leaky_relu(z: float, alpha=LEAKINESS_FACTOR):
        return np.where(z > 0, z, alpha * z)

class ActivationDerivative:
    def ident(_z: float):
        return 1
    def sigmoid(z: float):
        sigmoid_value = Activation.sigmoid(z)
        return sigmoid_value * (1.0 - sigmoid_value)
    def relu(z: float):
        np.where(z > 0, 1, 0)
    def leaky_relu(z: float, alpha=LEAKINESS_FACTOR):
        return np.where(z > 0, 1, alpha)
