from typing import Callable
from neural.losses.losses import Loss, LossDerivative
from neural.activations.activations import Activation, ActivationDerivative
from neural.initializers.initializers import Initializers


class MapActivation:
    def __init__(self, activation, activation_der, kernel_initializer):
        self.activation = activation
        self.activation_der = activation_der
        self.kernel_initializer = kernel_initializer

class MapLoss:
    def __init__(self, loss, loss_der):
        self.loss = loss
        self.loss_der = loss_der

map_activations: dict[str, Callable] = {
    'linear': MapActivation(Activation.ident, ActivationDerivative.ident, Initializers.random_normal),
    'sigmoid': MapActivation(Activation.sigmoid, ActivationDerivative.sigmoid, Initializers.xavier_normal),
    'relu': MapActivation(Activation.relu, ActivationDerivative.relu, Initializers.he),
    'leaky_relu': MapActivation(Activation.leaky_relu, ActivationDerivative.leaky_relu, Initializers.he)
}

map_loss: dict[str, Callable] = {
    'mean_squared_error': MapLoss(Loss.quadratic, LossDerivative.quadratic),
    'mse': MapLoss(Loss.quadratic, LossDerivative.quadratic),
    'mean_absolute_error': MapLoss(Loss.absolute, LossDerivative.absolute),
    'mae': MapLoss(Loss.absolute, LossDerivative.absolute),
    'binary_crossentropy': MapLoss(Loss.log, LossDerivative.log)
}
