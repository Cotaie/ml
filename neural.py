import numpy as np
from constants import BIAS_INPUT
from activations import ident, ident_der, map_activations, map_activations_der
from utils import _Utils, _ModelLayer, _Compile

class Layer:
    def __init__(self, units, activation=None, name=None):
        self._units = units
        self._activation = activation
        self._activation_der = self.get_activation_der()
        self._name = name

    def get_units(self):
        return self._units
    def get_activation(self):
        return _Utils.get_with_warning(map_activations, self._activation, ident, "activation function not found, linear is used")
    def get_activation_der(self):
        return _Utils.get(map_activations_der, self._activation, ident_der)
    def get_name(self, layer_no):
        return self._name if self._name is not None else f"layer_{layer_no}"

class Model:
    def __init__(self, model_arch, seed=None):
        self._optimizer = None
        self._loss = None
        self._loss_der = None
        self._model = self._create_model(model_arch, seed)
        self._gradient = self._init_gradient()

    def _create_model(self, model_arch, seed):
        if seed is not None:
            np.random.seed(seed)
        model = []
        no_layers_plus_input = len(model_arch)
        # create first model member (W matrix and activation function)
        first_layer_W = np.random.randn(model_arch[1].get_units(), model_arch[0] + BIAS_INPUT)
        first_layer_act_fct = model_arch[1].get_activation()
        first_layer_act_fct_der = model_arch[1].get_activation_der()
        first_layer_name = model_arch[1].get_name(1)
        model.append(_ModelLayer(first_layer_W, first_layer_act_fct, first_layer_act_fct_der, first_layer_name))
        # create remaining model members (W matrix and activation function)
        for cur_layer in range(2, no_layers_plus_input):
            cur_layer_W = np.random.randn(model_arch[cur_layer].get_units(), model_arch[cur_layer - 1].get_units() + BIAS_INPUT)
            cur_layer_act_fct = model_arch[cur_layer].get_activation()
            cur_layer_act_fct_der = model_arch[cur_layer].get_activation_der()
            cur_layer_name = model_arch[cur_layer].get_name(cur_layer)
            model.append(_ModelLayer(cur_layer_W, cur_layer_act_fct, cur_layer_act_fct_der, cur_layer_name))
        return model
    def _feed_forward(self, input):
        no_layers = len(self._model)
        bias_1 = np.array([BIAS_INPUT])
        output = input
        for cur_layer in range(no_layers):
            biased_input = np.concatenate((bias_1, output))
            output = self._model[cur_layer].get_activation()(self._model[cur_layer].get_matrix_W() @ biased_input)
        return output
    def _init_gradient(self):
        size = 0
        for cur_layer in self._model:
            size = size + cur_layer._matrix_W.size
        return np.empty(size)
    def fit(self, X, y):
        loss_C0 = self._loss(self._feed_forward(X[0]), y[0])
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