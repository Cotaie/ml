import numpy as np
from typing import Callable

def ident(z: float):
    return z

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def relu(z: float):
    return np.maximum(0.0, z)

map_activations: dict[str, Callable] = {
    'linear': ident,
    'sigmoid': sigmoid,
    'relu': relu
}

def ident_der(_z: float):
    return 1

def sigmoid_der(z: float):
    sigmoid_value = sigmoid(z)
    return sigmoid_value * (1.0 - sigmoid_value)

def relu_der(z: float):
    return 1.0 if z > 0 else 0.0

map_activations_der: dict[str, Callable] = {
    'linear': ident_der,
    'sigmoid': sigmoid_der,
    'relu': relu_der
}
