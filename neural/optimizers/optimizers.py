import numpy as np
from neural.constants import EPSILON

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._nesterov = nesterov
        self._velocity = {}

    def step(self, layer):
        lookahead = 0
        if self._nesterov and layer.name in self._velocity:
            lookahead = self._momentum * self._velocity[layer.name]
            layer.W -= lookahead  # Adjust weights with the lookahead term

        scaled_gradient = (1 - self._momentum) * layer.der_W
        self._velocity[layer.name] = self._momentum * self._velocity.get(layer.name, 0) + scaled_gradient

        layer.W += lookahead  # Reverse the lookahead adjustment (if lookahead is 0, this does nothing)
        layer.W -= self._learning_rate * self._velocity[layer.name]

        # layer.W -= self._learning_rate * layer.der_W

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
