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
        self._optimizer = None
        self._loss = None
        self._loss_der = None

    def _create_model(self, model_arch, seed: int | None):
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
    def _feed_forward(self, input, update_z: bool):
        output = input
        if update_z != False:
            for layer in self._model:
                z = layer.get_W() @ np.concatenate((BIAS_INPUT_NDARRAY, output))
                layer.set_z(z)
                output = layer.get_activation()(z)
            return output
        else:
            for layer in self._model:
                z = layer.get_W() @ np.concatenate((BIAS_INPUT_NDARRAY, output))
                output = layer.get_activation()(z)
            return output
    def _comp_loss_der_arr(self, output, y_i):
        loss_der_arr = []
        for j, out in enumerate(output):
            loss_der_arr.append(self._loss_der(out, y_i[j]))
        return loss_der_arr
    def _compute_gradient(output, hidden_layer):
        pass
    def _adjust_W(self):
        pass
    def compile(self, optimizer=None, loss=None):
        comp = _Compile(optimizer, loss)
        self._optimizer = optimizer
        self._loss = comp.get_loss()
        self._loss_der = comp.get_loss_der()
    def fit(self, X, Y):
        for index, x in enumerate(X):
            C_x = self._comp_loss_der_arr(self._feed_forward(x, True), Y[index])
            #print(f"loss der arr{index}",C_x)
            output = C_x
            no_layers = len(self._model)
            prev_layer = self._model[-2]
            #print("Prev_layer:", prev_layer._z)
            for curr in range(no_layers-1, 0, -1):
                self._model[curr]._der_W = output * prev_layer._activation(np.concatenate((BIAS_INPUT_NDARRAY, prev_layer._z)))
                print(f"index: {curr}:",self._model[curr]._der_W)
                #output = self._model[curr]._der_W
                prev_layer = self._model[curr]
            #print("test", self._model[1]._der_W)
            #hidden_layer._der_W = output * prev_layer._activation(np.concatenate((BIAS_INPUT_NDARRAY, prev_layer._z)))
            #print("derr, ",prev_layer._der_W)
            #print("teeest", self._model[0]._der_W)
            print("teat", output)
            for row in self._model[0]._der_W:
                row[:] = output * np.concatenate((BIAS_INPUT_NDARRAY, x))
                #print("row: ",row)
    def predict(self, input):
        return self._feed_forward(input, False)