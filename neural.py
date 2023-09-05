import numpy as np
from itertools import chain
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
        self._model = self._build(model_arch, seed)
        self._optimizer = None
        self._loss = None
        self._loss_der = None
        self._norm_fct = Normalization.no_normalization(None)
        self._learning_rate = 0.01
        self._reg = 0
        self._reg_fact = 1

    def _build(self, model_arch: list, seed: int | None):
        np.random.seed(seed)
        previous_layer = BasicLayer(model_arch[0])
        def _build_(layer, layer_index):
            nonlocal previous_layer
            model_layer = ModelLayer(previous_layer._units, layer._units, layer._activation, layer._kernel_initializer, layer.get_name_or_default(layer_index))
            previous_layer = layer
            return model_layer
        return [_build_(layer, layer_index) for layer_index, layer in enumerate(model_arch[1:], start=1)]
    def _feed_forward(self, input):
        output = input[:]
        def ff_update_z(layer):
            nonlocal output
            layer._z[:] = layer._W @ np.concatenate(([BIAS_INPUT], output))
            output = layer._activation(layer._z)
        for layer in self._model:
            ff_update_z(layer)
        return output
    def _adjust_W(self):
        self._reg = 0
        for layer in self._model:
            for row_w, row_der_w in zip(layer._W, layer._der_W):
                row_w[:] = [w - self._learning_rate * der_w for w, der_w in zip(row_w, row_der_w)]
                self._reg += sum(row_w)
    def _compute_gradients(self, output, x):
        layers_reversed = chain(reversed(self._model), iter([ModelFirstLayer(x, Activation.ident)]))
        prev_layer = next(layers_reversed, None)
        def _compute_gradients_(layer):
            nonlocal prev_layer
            nonlocal output
            prev_layer._der_W[:] = [out * layer._activation(np.concatenate(([BIAS_INPUT], layer._z))) for out in output]
            output = output @ prev_layer._W[:, 1:]
            prev_layer = layer
        for layer in layers_reversed:
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
            self._compute_gradients(self._loss_der(self._feed_forward(x_normed), np.array(y)), x_normed)
            self._adjust_W()
        for _ in range(epochs):
            for x, y in zip(X, Y):
                fit_(x, y)
    # def fit(self, X, Y, batch_size, epochs):
    #     #self._norm_fct = self._input_normalization(X)
    #     for _ in range(epochs):
    #         get_batch = Model._batch(X, Y, batch_size)
    #         for X_batch, Y_batch in get_batch:
    #             batch_loss_der = np.array([0.])
    #             for x, y in zip(X_batch, Y_batch):
    #                 x_normed = self._norm_fct(x)
    #                 #np.add(batch_loss_der, np.array(self._comp_loss_der_arr(self._feed_forward(x_normed, True), y)), out=batch_loss_der)
    #                 batch_loss_der += np.array(self._comp_loss_der_arr(self._feed_forward(x_normed, True), y))
    #             len_mse = len(Y_batch)
    #             #print("batch_size: ", len_mse)
    #             #print("abg_batch:", batch_loss_der)
    #             batch_loss_der /= float(len_mse)
    #             for x, y in zip(X_batch, Y_batch):
    #                 x_normed = self._norm_fct(x)
    #                 self._compute_gradients(batch_loss_der, x_normed)
    #                 self._adjust_W()
    def predict(self, input):
        return self._feed_forward(self._norm_fct(input))
