import numpy as np
import sys

def loss_quadratic(g, y_i):
    return np.square(g - y_i)

def loss_absolute(g, y_i):
    return np.abs(g - y_i)

def loss_log(g, y_i, eps=sys.float_info.epsilon):
    return -(y_i * np.log(g + eps) + (1 - y_i) * np.log(1 - g + eps))

func_map_loss = {
    'mean_squared_error': loss_quadratic,
    'mse': loss_quadratic,
    'mean_absolute_error': loss_absolute,
    'mae': loss_absolute,
    'binary_crossentropy': loss_log
}