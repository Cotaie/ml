import numpy as np
from typing import Callable
from constants import BIAS_INPUT, LOSS_NOT_FOUND
from activations import Activation
from normalization import Normalization
from utils import Utils, ModelLayer, ModelFirstLayer, BasicLayer, map_loss


class Layer(BasicLayer):
    def __init__(self, units: int, activation: str | None = None, kernel_initializer: Callable | None = None, name: str | None = None):
        super().__init__(units)
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._name = name

    def get_name_or_default(self, layer_no: int):
        return self._name if self._name is not None else f"layer_{layer_no}"

class Model:
    def __init__(self, model_arch: list, seed: int | None = None):
        self._model = self._create_model(model_arch, seed)
        self._optimizer = None
        self._loss = None
        self._loss_der = None
        self._norm_fct = Normalization.no_normalization(None)
        self._learning_rate = 0.01
        self._reg = 0
        self._reg_fact = 1

    def _create_model(self, model_arch: list, seed: int | None):
        np.random.seed(seed)
        previous_layer = BasicLayer(model_arch[0])
        def _create_model_(layer, layer_index):
            nonlocal previous_layer
            model_layer = ModelLayer(previous_layer._units, layer._units, layer._activation, layer._kernel_initializer, layer.get_name_or_default(layer_index))
            previous_layer = layer
            return model_layer
        return [_create_model_(layer, layer_index) for layer_index, layer in enumerate(model_arch[1:], start=1)]
    def _feed_forward(self, input, update_z: bool):
        output = input[:]
        def ff_update_z(layer):
            nonlocal output
            layer._z[:] = layer._W @ np.concatenate(([BIAS_INPUT], output))
            output = layer._activation(layer._z)
        def ff_no_update_z(layer):
            nonlocal output
            output = layer._activation(layer._W @ np.concatenate(([BIAS_INPUT], output)))
        ff = ff_update_z if update_z == True else ff_no_update_z
        for layer in self._model:
            ff(layer)
        return output
    def _comp_loss_der_arr(self,y_pred, y_real):
        return [self._loss_der(y_i_pred, y_i_real) for y_i_pred, y_i_real in zip(y_pred, y_real)]
    def _adjust_W(self):
        self._reg = 0
        for layer in self._model:
            for row_w, row_der_w in zip(layer._W, layer._der_W):
                row_w[:] = [w - self._learning_rate * der_w for w, der_w in zip(row_w, row_der_w)]
                self._reg += sum(row_w)
    def _compute_gradients(self, output, x):
        layers_reversed = self._model[::-1]
        layers_reversed.append(ModelFirstLayer(x, Activation.ident))
        prev_layer = layers_reversed[0]
        def _compute_gradients_(layer):
            nonlocal prev_layer
            nonlocal output
            prev_layer._der_W[:] = [out * layer._activation(np.concatenate(([BIAS_INPUT], layer._z))) for out in output]
            output = output @ prev_layer._W[:, 1:]
            prev_layer = layer
        for layer in layers_reversed[1:]:
            _compute_gradients_(layer)
    def compile(self, optimizer=None, loss=None, input_normalization=None):
        self._optimizer = optimizer
        self._loss = Utils.get_with_warning(map_loss, loss, map_loss['mean_squared_error'], LOSS_NOT_FOUND)._loss
        self._loss_der = Utils.get(map_loss, loss, map_loss['mean_squared_error'])._loss_der
        self._input_normalization = Normalization.no_normalization if input_normalization is None else input_normalization
    def fit(self, X, Y, batch_size=32, epochs=1):
        self._norm_fct = self._input_normalization(X)
        def fit_(x, y):
            x_normed = self._norm_fct(x)
            self._compute_gradients(self._comp_loss_der_arr(self._feed_forward(x_normed, True), y), x_normed)
            self._adjust_W()
        for _ in range(epochs):
            for x, y in zip(X, Y):
                fit_(x, y)
    def predict(self, input):
        return self._feed_forward(self._norm_fct(input), False)
