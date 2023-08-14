import numpy as np
from constants import BIAS_INPUT
from activations import ident, func_map

class CreateLayer:
    def __init__(self, units, activation=None, name=None):
        self.units = units
        self.activation = activation
        self.name = name
    def get_units(self):
        return self.units
    def get_activation(self):
        activation_str = 'linear' if self.activation is None else self.activation
        return activation_str
    def get_name(self, layer_no):
        return self.name if self.name is not None else f"layer_{layer_no}"

class Layer:
    def __init__(self, matrix_W, activation, name):
        self.matrix_W = matrix_W
        self.activation = activation
        self.name = name
    def get_matrix_W(self):
        return self.matrix_W
    def get_activation(self):
        return self.activation
    def get_name(self):
        return self.name

class Model(list):

    def __init__(self, model_arch, seed=None):
        self.optimizer = None
        self.loss = None
        self.model = self._create_model(model_arch, seed)

    def _create_model(self, model_arch, seed):
        if seed is not None:
            np.random.seed(seed)
        self.model = []
        no_layers_plus_input = len(model_arch)
        # create first model member (W matrix and activation function)
        first_layer_W = np.random.randn(model_arch[1].get_units(), model_arch[0] + BIAS_INPUT)
        first_layer_act_fct = func_map.get(model_arch[1].get_activation(),(lambda: ident)() )
        first_layer_name = model_arch[1].get_name(1)
        self.model.append(Layer(first_layer_W, first_layer_act_fct, first_layer_name))
        # create remaining model members (W matrix and activation function)
        for cur_layer in range(2, no_layers_plus_input):
            cur_layer_W = np.random.randn(model_arch[cur_layer].get_units(), model_arch[cur_layer - 1].get_units() + BIAS_INPUT)
            cur_layer_act_fct = func_map.get(model_arch[cur_layer].get_activation(),(lambda: ident)() )
            cur_layer_name = model_arch[cur_layer].get_name(cur_layer)
            self.model.append(Layer(cur_layer_W, cur_layer_act_fct, cur_layer_name))
        return self.model
    
    def _feed_forward(self, input):
        no_layers = len(self.model)
        bias_1 = np.array([BIAS_INPUT])

        output = input
        for cur_layer in range(no_layers):
            biased_input = np.concatenate((bias_1, output))
            output = self.model[cur_layer].get_activation()(self.model[cur_layer].get_matrix_W() @ biased_input)
        return output
    
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def predict(self, input):
        return self._feed_forward(input)