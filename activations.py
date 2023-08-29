import numpy as np
from typing import Callable

def ident(z: float):
    return z

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -709, 709)))

def relu(z: float):
    return np.maximum(0.0, z)

def leaky_relu(z: float, alpha=0.01):
    np.where(z > 0, z, alpha * z)

map_activations: dict[str, Callable] = {
    'linear': ident,
    'sigmoid': sigmoid,
    'relu': relu,
    'leaky_relu': leaky_relu
}

def ident_der(_z: float):
    return 1

def sigmoid_der(z: float):
    sigmoid_value = sigmoid(z)
    return sigmoid_value * (1.0 - sigmoid_value)

def relu_der(z: float):
    return 1.0 if z > 0 else 0.0

def leaky_rely_der(z: float, alpha=0.01):
    return np.where(z > 0, 1, alpha)

map_activations_der: dict[str, Callable] = {
    'linear': ident_der,
    'sigmoid': sigmoid_der,
    'relu': relu_der,
    'leaky_relu': leaky_rely_der
}
