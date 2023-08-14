import numpy as np
from constants import BIAS_INPUT
from activ import ident, func_map

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

def create_model(*args):
    model_arch = args[0]
    if len(args) > 1:
      seed = args[1]
      np.random.seed(seed)
    model = []
    no_layers_plus_input = len(model_arch)
    # create first model member (W matrix and activation function)
    first_layer_W = np.random.randn(model_arch[1].get_units(), model_arch[0] + BIAS_INPUT)
    first_layer_act_fct = func_map.get(model_arch[1].get_activation(),(lambda: ident)() )
    first_layer_name = model_arch[1].get_name(1)
    model.append(Layer(first_layer_W, first_layer_act_fct, first_layer_name))
    # create remaining model members (W matrix and activation function)
    for cur_layer in range(2, no_layers_plus_input):
        cur_layer_W = np.random.randn(model_arch[cur_layer].get_units(), model_arch[cur_layer - 1].get_units() + BIAS_INPUT)
        cur_layer_act_fct = func_map.get(model_arch[cur_layer].get_activation(),(lambda: ident)() )
        cur_layer_name = model_arch[cur_layer].get_name(cur_layer)
        model.append(Layer(cur_layer_W, cur_layer_act_fct, cur_layer_name))
    return model

def feed_forward(model):
    no_layers = len(model)
    bias_1 = np.array([BIAS_INPUT])

    def feed_forward_(input):
        output = input
        for cur_layer in range(no_layers):
            biased_input = np.concatenate((bias_1, output))
            output = model[cur_layer].get_activation()(model[cur_layer].get_matrix_W() @ biased_input)
        return output

    return feed_forward_
