import numpy as np
import warnings
from constants import MEAN, VARIANCE, LOSS_NOT_FOUND, ACTIVATION_NOT_FOUND
from loss import loss_quadratic, loss_quadratic_der, map_loss, map_loss_der
from typing import Callable
from activations import ident, ident_der, map_activations, map_activations_der

class _Utils:
      def get_with_warning(dict: dict, key: str, default: Callable, warning: str):
          if key not in dict:
             warnings.warn(warning)
          return dict.get(key, default)
      def get(dict, key, default) -> Callable:
          return dict.get(key, default)

class ModelLayer:
    def __init__(self, no_inputs: int, no_neurons: int, activation: str | None, name: str | None):
        self._W = np.hstack((np.zeros((no_neurons, 1)),np.random.normal(MEAN, VARIANCE, (no_neurons, no_inputs))))
        self._der_W = []
        self._z = None
        self._activation = _Utils.get_with_warning(map_activations, activation, ident, ACTIVATION_NOT_FOUND)
        self._activation_der = _Utils.get(map_activations_der, activation, ident_der)
        self._name = name

    @property
    def W(self) -> np.ndarray:
        return self._W
    @property
    def z(self):
        return self._z
    @z.setter
    def z(self, z):
        self._z = z
    @property
    def activation(self) -> Callable:
        return self._activation
    @property
    def name(self) -> str:
        return self._name
    @property
    def der_W(self):
        return self._der_W
    @der_W.setter
    def der_W(self, der_W):
        self._der_W = der_W

class Compile:
    def __init__(self, optimizer: str | None = None, loss: str | None = None):
        self._optimizer = optimizer
        self._loss = loss

    def get_loss(self):
        return _Utils.get_with_warning(map_loss, self._loss, loss_quadratic, LOSS_NOT_FOUND)
    def get_loss_der(self):
        return _Utils.get(map_loss_der, self._loss, loss_quadratic_der)

class BasicLayer:
    def __init__(self, units: int):
        self._units = units

    @property
    def units(self):
        return self._units