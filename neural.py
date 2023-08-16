import numpy as np
from constants import BIAS_INPUT, BIAS_INPUT_NDARRAY, ACTIVATION_NOT_FOUND
from activations import ident, ident_der, map_activations, map_activations_der
from utils import _Utils, _ModelLayer, _Compile, _BasicLayer
from typing import Callable

class Layer(_BasicLayer):
    def __init__(self, units: int, activation: Callable | None = None, name: str| None = None):
        super().__init__(units)
        self._activation = self._get_activation(activation)
        self._activation_der = self._get_activation_der(activation)
        self._name = name

    def _get_activation(self, activation):
        return _Utils.get_with_warning(map_activations, activation, ident, ACTIVATION_NOT_FOUND)
    def _get_activation_der(self, activation):
        return _Utils.get(map_activations_der, activation, ident_der)
    def get_activation(self):
        return self._activation
    def get_activation_der(self):
        return self._activation_der
    def get_name(self, layer_no: int):
        return self._name if self._name is not None else f"layer_{layer_no}"

class Model:
    def __init__(self, model_arch: list, seed: int | None = None):
        self._model = self._create_model(model_arch, seed)
        self._gradient = self._init_gradient()
        self._optimizer = None
        self._loss = None
        self._loss_der = None

    def _create_model(self, model_arch, seed: int | None):
        if seed is not None:
            np.random.seed(seed)
        model = []
        previous_layer = _BasicLayer(model_arch[0])
        for layer_index, layer in enumerate(model_arch[1:], start=1):
            model.append(_ModelLayer(np.random.randn(layer.get_units(), previous_layer.get_units() + BIAS_INPUT),
                                     layer.get_activation(),
                                     layer.get_activation_der(),
                                     layer.get_name(layer_index)))
            previous_layer = layer
        return model
    def _init_gradient(self):
        size = 0
        for layer in self._model:
            size = size + layer.get_W_size()
        return np.empty(size)
    def _feed_forward(self, input):
        output = input
        for layer in self._model:
            z = layer.get_W() @ np.concatenate((BIAS_INPUT_NDARRAY, output))
            layer.set_z(z)
            output = layer.get_activation()(z)
        return output
    def fit(self, X, y):
        #loss_C0 = self._loss(self._feed_forward(X[0]), y[0])
        pass
    def _compute_gradient(self):
        pass
    def compile(self, optimizer=None, loss=None):
        comp = _Compile(optimizer, loss)
        self._optimizer = optimizer
        self._loss = comp.get_loss()
        self._loss_der = comp.get_loss_der()
    def train(self):
        pass
    def predict(self, input):
        return self._feed_forward(input)