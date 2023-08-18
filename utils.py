import numpy as np
import warnings
from constants import LOSS_NOT_FOUND
from loss import loss_quadratic, loss_quadratic_der, map_loss, map_loss_der
from typing import Callable

class _Utils:
      def get_with_warning(dict: dict, key: str, default: Callable, warning: str):
          if key not in dict:
             warnings.warn(warning)
          return dict.get(key, default)
      def get(dict, key, default) -> Callable:
          return dict.get(key, default)

class _ModelLayer:
    def __init__(self, W: np.ndarray, activation: Callable, activation_der: Callable, name: str):
        self._W = W
        self._der_W = np.ones(np.shape(self._W))
        self._z = None
        self._activation = activation
        self._activation_der = activation_der
        self._name = name

    def get_W(self) -> np.ndarray:
        return self._W
    def get_activation(self) -> Callable:
        return self._activation
    def get_name(self) -> str:
        return self._name
    def get_z(self):
        return self._z
    def set_z(self, output):
        self._z = output

class _Compile:
    def __init__(self, optimizer: str | None = None, loss: str | None = None):
        self._optimizer = optimizer
        self._loss = loss

    def get_loss(self):
        return _Utils.get_with_warning(map_loss, self._loss, loss_quadratic, LOSS_NOT_FOUND)
    def get_loss_der(self):
        return _Utils.get(map_loss_der, self._loss, loss_quadratic_der)

class _BasicLayer:
    def __init__(self, units: int):
        self._units = units
    def get_units(self):
        return self._units