import numpy as np
import warnings
from typing import Callable
from constants import ACTIVATION_NOT_FOUND
from loss import Loss, LossDerivative
from activations import Activation, ActivationDerivative
from initializers import Initializers


class BasicLayer:
    def __init__(self, units: int):
        self._units = units

class ModelFirstLayer:
    def __init__(self, z, activation):
        self._z = z
        self._activation = activation

class MapActivation:
    def __init__(self, activation, activation_der, kernel_initializer):
        self._activation = activation
        self._activation_der = activation_der
        self._kernel_initializer = kernel_initializer

class MapLoss:
    def __init__(self, loss, loss_der):
        self._loss = loss
        self._loss_der = loss_der

class Utils:
      def get_with_warning(dict: dict, key: str, default: Callable, warning: str):
          if key not in dict:
             warnings.warn(warning)
          return dict.get(key, default)
      def get(dict, key, default) -> Callable:
          return dict.get(key, default)

class ModelLayer:
    def __init__(self, no_inputs: int, no_neurons: int, activation: str | None, kernel_initializer: Callable | None, name: str | None):
        self._activation = Utils.get_with_warning(map_activations, activation, map_activations['linear'], ACTIVATION_NOT_FOUND)._activation
        self._activation_der = Utils.get(map_activations, activation,  map_activations['linear'])._activation_der
        self._W = Utils.get(map_activations, self._activation,  map_activations['linear'])._kernel_initializer(no_inputs, no_neurons) if kernel_initializer is None else kernel_initializer(no_inputs, no_neurons)
        self._der_W = np.empty(self._W.shape)
        self._z = np.empty(no_neurons)
        self._name = name

map_activations: dict[str, Callable] = {
    'linear': MapActivation(Activation.ident, ActivationDerivative.ident, Initializers.random_normal),
    'sigmoid': MapActivation(Activation.sigmoid, ActivationDerivative.sigmoid, Initializers.xavier_normal),
    'relu': MapActivation(Activation.relu, ActivationDerivative.relu, Initializers.he),
    'leaky_relu': MapActivation(Activation.leaky_relu, ActivationDerivative.leaky_relu, Initializers.he)
}

map_loss: dict[str, Callable] = {
    'mean_squared_error': MapLoss(Loss.quadratic, LossDerivative.quadratic),
    'mse': MapLoss(Loss.quadratic, LossDerivative.quadratic),
    'mean_absolute_error': MapLoss(Loss.absolute, LossDerivative.absolute),
    'mae': MapLoss(Loss.absolute, LossDerivative.absolute),
    'binary_crossentropy': MapLoss(Loss.log, LossDerivative.log)
}
