import numpy as np
from constants import BIAS_INPUT
from activations import ident, func_map_activations
from loss import loss_quadratic, func_map_loss

class Layer:
    def __init__(self, units, activation=None, name=None):
        self._units = units
        self._activation = activation
        self._name = name

    def get_units(self):
        return self._units
    
    def get_activation(self):
        return func_map_activations.get('linear' if self._activation is None else self._activation, (lambda: ident)())
    
    def get_name(self, layer_no):
        return self._name if self._name is not None else f"layer_{layer_no}"

class Model:
    def __init__(self, model_arch, seed=None):
        self._optimizer = None
        self._loss = None
        self._model = self._create_model(model_arch, seed)

    class _ModelLayer:
        def __init__(self, matrix_W, activation, name):
            self._matrix_W = matrix_W
            self._activation = activation
            self._name = name
        def get_matrix_W(self):
            return self._matrix_W
        def get_activation(self):
            return self._activation
        def get_name(self):
            return self._name
    
    class _Compile:
        def __init__(self, optimizer=None, loss=None):
            self._optimizer = optimizer
            self._loss = loss
        def get_loss(self):
            return func_map_loss.get('mse' if self._loss is None else self._loss, (lambda: loss_quadratic)())

    def _create_model(self, model_arch, seed):
        if seed is not None:
            np.random.seed(seed)
        self.model = []
        no_layers_plus_input = len(model_arch)
        # create first model member (W matrix and activation function)
        first_layer_W = np.random.randn(model_arch[1].get_units(), model_arch[0] + BIAS_INPUT)
        first_layer_act_fct = model_arch[1].get_activation()
        first_layer_name = model_arch[1].get_name(1)
        self.model.append(self._ModelLayer(first_layer_W, first_layer_act_fct, first_layer_name))
        # create remaining model members (W matrix and activation function)
        for cur_layer in range(2, no_layers_plus_input):
            cur_layer_W = np.random.randn(model_arch[cur_layer].get_units(), model_arch[cur_layer - 1].get_units() + BIAS_INPUT)
            cur_layer_act_fct = model_arch[cur_layer].get_activation()
            cur_layer_name = model_arch[cur_layer].get_name(cur_layer)
            self.model.append(self._ModelLayer(cur_layer_W, cur_layer_act_fct, cur_layer_name))
        return self.model
    
    def _feed_forward(self, input):
        no_layers = len(self.model)
        bias_1 = np.array([BIAS_INPUT])

        output = input
        for cur_layer in range(no_layers):
            biased_input = np.concatenate((bias_1, output))
            output = self.model[cur_layer].get_activation()(self.model[cur_layer].get_matrix_W() @ biased_input)
        return output
    
    def compile(self, optimizer=None, loss=None):
        comp = self._Compile(optimizer, loss)
        self._optimizer = optimizer
        self._loss = comp.get_loss()
    
    def train(self):
        pass

    def predict(self, input):
        return self._feed_forward(input)