import numpy as np
from neural.constants import EPSILON

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, clipvalue=None):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._nesterov = nesterov
        self._velocity_W = {}
        self._velocity_b = {}
        self._clipvalue = clipvalue

    def step(self, layer):
        # # For weights (W)
        # lookahead_W = 0
        # if self._nesterov and layer.name in self._velocity_W:
        #     lookahead_W = self._momentum * self._velocity_W[layer.name]
        #     layer.W -= lookahead_W  # Adjust weights with the lookahead term

        # scaled_gradient_W = (1 - self._momentum) * layer.der_W
        # self._velocity_W[layer.name] = self._momentum * self._velocity_W.get(layer.name, 0) + scaled_gradient_W

        # layer.W += lookahead_W  # Reverse the lookahead adjustment (if lookahead_W is 0, this does nothing)
        # layer.W -= self._learning_rate * self._velocity_W[layer.name]

        # # For biases (b)
        # lookahead_b = 0
        # if self._nesterov and layer.name in self._velocity_b:
        #     lookahead_b = self._momentum * self._velocity_b[layer.name]
        #     layer.b -= lookahead_b  # Adjust biases with the lookahead term

        # scaled_gradient_b = (1 - self._momentum) * layer.der_b
        # self._velocity_b[layer.name] = self._momentum * self._velocity_b.get(layer.name, 0) + scaled_gradient_b

        # layer.b += lookahead_b  # Reverse the lookahead adjustment (if lookahead_b is 0, this does nothing)
        # layer.b -= self._learning_rate * self._velocity_b[layer.name]

        if self._clipvalue is not None:
            clipvalue = self._clipvalue
            np.clip(layer.der_b, a_min=-clipvalue, a_max=clipvalue, out=layer.der_b)
            np.clip(layer.der_W, a_min=-clipvalue, a_max=clipvalue, out=layer.der_W)
        layer.b -= self._learning_rate * layer.der_b
        layer.W -= self._learning_rate * layer.der_W

class AdaGrad:
    def __init__(self, learning_rate=0.01, epsilon=EPSILON):
        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self._G = {}  # Stores the running sum of squares of gradients for each layer

    def step(self, layer):
        if layer.name not in self._G:
            self._G[layer.name] = np.zeros_like(layer.W)

        # Update running sum of squares of gradients
        self._G[layer.name] += layer.der_W ** 2

        # Update weights using AdaGrad rule
        layer.W -= self._learning_rate / (np.sqrt(self._G[layer.name]) + self._epsilon) * layer.der_W
