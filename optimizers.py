class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self._learning_rate = learning_rate
        self._momentum = momentum

    def step(self, weights, gradient):
        weights -= self._learning_rate * gradient