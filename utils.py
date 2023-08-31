import numpy as np
import warnings
from typing import Callable
from constants import ACTIVATION_POS, ACTIVATION_DER_POS, WEIGHTS_INIT_POS, LOSS_POS, LOSS_DER_POS, LOSS_NOT_FOUND, ACTIVATION_NOT_FOUND
from loss import Loss, LossDerivative
from activations import Activation, ActivationDerivative
from initializers import Initializers


map_activations: dict[str, Callable] = {
    'linear': (Activation.ident, ActivationDerivative.ident, Initializers.random_normal),
    'sigmoid': (Activation.sigmoid, ActivationDerivative.sigmoid, Initializers.xavier_normal),
    'relu': (Activation.relu, ActivationDerivative.relu, Initializers.he),
    'leaky_relu': (Activation.leaky_relu, ActivationDerivative.leaky_relu, Initializers.he)
}

map_loss: dict[str, Callable] = {
    'mean_squared_error': (Loss.quadratic, LossDerivative.quadratic),
    'mse': (Loss.quadratic, LossDerivative.quadratic),
    'mean_absolute_error': (Loss.absolute, LossDerivative.absolute),
    'mae': (Loss.absolute, LossDerivative.absolute),
    'binary_crossentropy': (Loss.log, LossDerivative.log)
}

class _Utils:
      def get_with_warning(dict: dict, key: str, default: Callable, warning: str):
          if key not in dict:
             warnings.warn(warning)
          return dict.get(key, default)
      def get(dict, key, default) -> Callable:
          return dict.get(key, default)

class ModelLayer:
    def __init__(self, no_inputs: int, no_neurons: int, activation: str | None, kernel_initializer: Callable | None, name: str | None):
        self._activation = _Utils.get_with_warning(map_activations, activation, (Activation.ident, None, None), ACTIVATION_NOT_FOUND)[ACTIVATION_POS]
        self._activation_der = _Utils.get(map_activations, activation, (None, ActivationDerivative.ident, None))[ACTIVATION_DER_POS]
        self._W = self._init_kernel(no_inputs, no_neurons, kernel_initializer)
        self._der_W = []
        self._z = None
        self._name = name

    def _init_kernel(self,no_inputs, no_neurons, kernel_initializer):
        if kernel_initializer is None:
            return _Utils.get(map_activations, self._activation, (None, None, Initializers.random_normal))[WEIGHTS_INIT_POS](no_inputs, no_neurons)
        else:
            return kernel_initializer(no_inputs, no_neurons)
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
        return _Utils.get_with_warning(map_loss, self._loss, (Loss.quadratic, None), LOSS_NOT_FOUND)[LOSS_POS]
    def get_loss_der(self):
        return _Utils.get(map_loss, self._loss, (None, LossDerivative.quadratic))[LOSS_DER_POS]

class BasicLayer:
    def __init__(self, units: int):
        self._units = units

    @property
    def units(self):
        return self._units
