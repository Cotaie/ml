import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._nesterov = nesterov
        self._velocity = {}

    def step(self, layer):
        if layer.name not in self._velocity:
            self._velocity[layer.name] = np.zeros_like(layer.W)
        velocity = self._momentum * self._velocity[layer.name] + (1 - self._momentum) * layer.der_W
        layer.W -= self._learning_rate * velocity
        self._velocity[layer.name] = velocity
        #layer.W -= self._learning_rate * layer.der_W