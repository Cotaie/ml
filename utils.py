import warnings
from loss import loss_quadratic, loss_quadratic_der, map_loss, map_loss_der

class _Utils:
      def get_with_warning(dict, key, default, warning):
          if key not in dict:
             warnings.warn(warning)
          return dict.get(key, default)
      def get(dict, key, default):
          return dict.get(key, default)

class _ModelLayer:
    def __init__(self, matrix_W, activation, activation_der, name):
        self._matrix_W = matrix_W
        self._activation = activation
        self._activation_der = activation_der
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
        return _Utils.get_with_warning(map_loss, self._loss, loss_quadratic, "loss function not found, mse is used")
    def get_loss_der(self):
        return _Utils.get(map_loss_der, self._loss, loss_quadratic_der)