import numpy as np
from constants import ACT_FCT, NO_NEURONS, BIAS_INPUT, PAR_W

def create_model(*args):
    model_arch = args[0]
    if len(args) > 1:
      seed = args[1]
      np.random.seed(seed)
    else:
       np.random.seed()
    model = []
    no_layers_plus_input = np.shape(model_arch)[0]
    # create first model member (W matrix and activation function)
    no_inputs = np.shape(model_arch[0])[0]
    first_layer_W = np.random.randn(model_arch[1][NO_NEURONS], no_inputs + BIAS_INPUT)
    first_layer_act_fct = model_arch[1][ACT_FCT]
    model.append((first_layer_W, first_layer_act_fct))
    # create remaining model members (W matrix and activation function)
    for i in range(2, no_layers_plus_input):
        no_inputs = model_arch[i - 1][NO_NEURONS]
        cur_layer_W = np.random.randn(model_arch[i][NO_NEURONS], no_inputs + BIAS_INPUT)
        cur_layer_act_fct = model_arch[i][ACT_FCT]
        model.append((cur_layer_W, cur_layer_act_fct))
    return model

def feed_forward(model):
    no_layers = len(model)
    bias_1 = np.array([BIAS_INPUT])

    def feed_forward_(input):
        output = input
        for layer in range(no_layers):
            biased_input = np.concatenate((bias_1, output))
            output = model[layer][ACT_FCT](model[layer][PAR_W] @ biased_input)
        return output

    return feed_forward_
