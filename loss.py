import numpy as np
from constants import EPS
from typing import Callable

def loss_quadratic(g: float, y_i: float) -> float:
    return np.square(g - y_i)

def loss_absolute(g: float, y_i: float):
    return np.abs(g - y_i)

def loss_log(g: float, y_i: float, eps=EPS) -> float:
    return -(y_i * np.log(g + eps) + (1 - y_i) * np.log(1 - g + eps))

map_loss: dict[str, Callable] = {
    'mean_squared_error': loss_quadratic,
    'mse': loss_quadratic,
    'mean_absolute_error': loss_absolute,
    'mae': loss_absolute,
    'binary_crossentropy': loss_log
}

def loss_quadratic_der(g, y_i):
    return 2 * (g - y_i)

def loss_absolute_der(g, y_i):
    return np.where(g > y_i, 1, np.where(g < y_i, -1, 0))

def loss_log_der(g, y_i):
    return -y_i / g + (1 - y_i) / (1 - g)

map_loss_der: dict[str, Callable] = {
    'mean_squared_error': loss_quadratic_der,
    'mse': loss_quadratic_der,
    'mean_absolute_error': loss_absolute_der,
    'mae': loss_absolute_der,
    'binary_crossentropy': loss_log_der
}