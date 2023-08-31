import numpy as np
from constants import BIAS_INPUT_NDARRAY
from activations import ident
from utils import ModelLayer, Compile, BasicLayer
from typing import Callable

class Layer(BasicLayer):
    def __init__(self, units: int, activation: str | None = None, kernel_initializer: Callable | None = None, name: str | None = None):
        super().__init__(units)
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._name = name

    @property
    def activation(self):
        return self._activation
    @property
    def kernel_initializer(self):
        return self._kernel_initializer
    def get_name_or_default(self, layer_no: int):
        return self._name if self._name is not None else f"layer_{layer_no}"

class Model:
    def __init__(self, model_arch: list, seed: int | None = None):
        self._model = self._create_model(model_arch, seed)
        self._optimizer = None
        self._loss = None
        self._loss_der = None
        self._learning_rate = 0.01
        self._reg = 0
        self._reg_fact = 1

    def _create_model(self, model_arch: list, seed: int | None):
        np.random.seed(seed)
        model = []
        previous_layer = BasicLayer(model_arch[0])
        for layer_index, layer in enumerate(model_arch[1:], start=1):
            model.append(ModelLayer(previous_layer.units, layer.units, layer.activation, layer.kernel_initializer, layer.get_name_or_default(layer_index)))
            previous_layer = layer
        return model
    def _feed_forward(self, input, update_z: bool):
        output = input
        if update_z != False:
            for layer in self._model:
                z = layer.W @ np.concatenate((BIAS_INPUT_NDARRAY, output))
                layer.z = z
                output = layer.activation(z)
            return output
        else:
            for layer in self._model:
                z = layer.W @ np.concatenate((BIAS_INPUT_NDARRAY, output))
                output = layer.activation(z)
            return output
    def _comp_loss_der_arr(self,y_pred, y_real):
        loss_der_arr = []
        for y_i_pred, y_i_real in zip(y_pred, y_real):
            loss_der_arr.append(self._loss_der(y_i_pred, y_i_real))
        return loss_der_arr
    def _compute_neuron_W_der(out, prev_activation, prev_z):
        return out * prev_activation(np.concatenate((BIAS_INPUT_NDARRAY, prev_z)))
    def _adjust_W(self):
        self._reg = 0
        for layer in self._model:
            for row_w, row_der_w in zip(layer.W, layer.der_W):
                row_w[:] = [w - self._learning_rate * der_w for w, der_w in zip(row_w, row_der_w)]
                self._reg += sum(row_w)
    def _compute_gradients(self, output, x):
        for i in range(len(self._model)-1, 0, -1):
            self._model[i].der_W.clear()
            for out in output:
                self._model[i].der_W.append(Model._compute_neuron_W_der(out, self._model[i-1].activation, self._model[i-1].z))
            output = output @ self._model[i].W[:,1:]
        first_layer = self._model[0]
        first_layer.der_W.clear()
        for out in output:
            first_layer.der_W.append(Model._compute_neuron_W_der(out, ident, x))
    def compile(self, optimizer=None, loss=None):
        comp = Compile(optimizer, loss)
        self._optimizer = optimizer
        self._loss = comp.get_loss()
        self._loss_der = comp.get_loss_der()
    def fit(self, X, Y, epochs=1):
        for _ in range(epochs):
            for x, y in zip(X, Y):
                self._compute_gradients(self._comp_loss_der_arr(self._feed_forward(x, True), y), x)
                self._adjust_W()
    def predict(self, input):
        if self._feed_forward(input, False)[0] < 0.5:
            return 0.
        else:
            return 1.