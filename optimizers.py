import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._nesterov = nesterov
        self._previous_update = None

    def step(self, weights, gradient):
        # # If the previous update is not initialized, initialize it to zeros (same shape as weights)
        # if self._previous_update is None:
        #     self._previous_update = np.zeros_like(weights)

        # # If using Nesterov momentum, compute the gradient after the "look-ahead"
        # if self._nesterov:
        #     lookahead_gradient = gradient + self._momentum * self._previous_update
        #     update = self._momentum * self._previous_update - self._learning_rate * lookahead_gradient
        # else:
        #     update = self._momentum * self._previous_update - self._learning_rate * gradient

        # weights += update

        # # Store the current update for the next iteration
        # self._previous_update = update
        weights -= self._learning_rate * gradient