import numpy as np
import warnings
from typing import Callable
from neural.normalizations import Normalization
from neural.optimizers import SGD
from neural.evaluations import Evaluate
from neural.model.maps import MapActivation, MapLoss, map_loss, map_activations
from neural._basic import BasicLayer
from neural.constants import LOSS_NOT_FOUND, ACTIVATION_NOT_FOUND


class Model:
    def __init__(self, model_arch: list, seed: int | None = None):
        self._layers = self._build(model_arch, seed)
        self._optimizer = None
        self._loss = None
        self._loss_der = None
        self._norm_fct = Normalization.no_normalization(None)
        self._reg = 0
        self._reg_fact = 1
        self._clip_W = False
    @staticmethod
    def get_with_warning(dict: dict, key: str, default: MapActivation | MapLoss , warning: str):
        """
        """
        if key not in dict:
            warnings.warn(warning)
        return dict.get(key, default)
    @staticmethod
    def get(dict: dict, key: str, default: MapActivation | MapLoss) -> Callable:
        """
        """
        return dict.get(key, default)
    @staticmethod
    def _batch(input, output, batch_size):
        """
        """
        num_batches = (len(input) + batch_size - 1) // batch_size
        for i in range(num_batches):
            yield (input[i * batch_size:(i + 1) * batch_size], output[i * batch_size:(i + 1) * batch_size])
    @staticmethod
    def _clip_der(der, value=1):
        """
        """
        np.clip(der, a_min=-value, a_max=value, out=der)
    @staticmethod
    def _normalize_der_W(der_b, der_W, norm_value=1.0):
        """
        """
        der_bW = np.hstack((der_b.reshape(-1,1), der_W))
        norm = np.linalg.norm(der_bW)
        #print("norm: ", norm)
        if norm == 0:
            return (np.array([sub_lst[0] for sub_lst in der_bW]), np.array([sub_lst[1:] for sub_lst in der_bW]))
        else:
            return ((np.array([sub_lst[0] for sub_lst in der_bW]) / norm) * norm_value, (np.array([sub_lst[1:] for sub_lst in der_bW]) / norm) * norm_value)

    def _build(self, model_arch: list, seed: int | None):
        """
        """
        np.random.seed(seed)
        previous_layer = BasicLayer(model_arch[0])
        def _build_layer(layer, layer_index):
            nonlocal previous_layer
            model_layer = Model._ModelLayer(previous_layer.units, layer.units, layer.activation, layer.kernel_initializer, layer.get_name_or_default(layer_index))
            previous_layer = layer
            return model_layer
        return [_build_layer(layer, layer_index) for layer_index, layer in enumerate(model_arch[1:], start=1)]

    def _feedforward(self, input, update_z=False):
        """
        """
        output = input[:]
        for layer in self._layers:
            output = output @ layer.W.T + layer.b
            if update_z:
                layer.z[:] = output
            output = layer.activation(output)
        return output

    def _update_W(self):
        """
        """
        for layer in self._layers:
            self._optimizer.step(layer)

    def _backpropagation(self,pred_y, y, x):
        """
        """
        layers_reversed = list(reversed(self._layers))
        prev_layer = layers_reversed[0]
        prev_layer.delta[:] = self._loss_der(pred_y, np.array(y)) * prev_layer.activation_der(prev_layer.z)
        for layer in layers_reversed[1:]:
            prev_layer.der_b[:] = prev_layer.delta
            prev_layer.der_W[:] = prev_layer.delta * layer.activation(layer.z)
            if (False):
                # Model._clip_der(prev_layer.der_b)
                # Model._clip_der(prev_layer.der_W)
                prev_layer.der_b, prev_layer.der_W = Model._normalize_der_W(prev_layer.der_b, prev_layer.der_W)
            layer.delta = (prev_layer.W.T @ prev_layer.delta) * layer.activation_der(layer.z)
            prev_layer = layer
        layers_reversed[-1].der_b[:] = prev_layer.delta
        layers_reversed[-1].der_W[:] = [delta * x for delta in prev_layer.delta]

    def compile(self, optimizer=SGD(), loss=None, input_normalization=None):
        """
        """
        self._optimizer = optimizer
        self._loss = Model.get_with_warning(map_loss, loss, map_loss['mean_squared_error'], LOSS_NOT_FOUND).loss
        self._loss_der = Model.get(map_loss, loss, map_loss['mean_squared_error']).loss_der
        self._input_normalization = Normalization.no_normalization if input_normalization is None else input_normalization

    def fit(self, X, Y, batch_size=32, epochs=1):
        """
        """
        X = np.array(X)
        Y = np.array(Y)
        self._norm_fct = self._input_normalization(X)
        nr_examples = len(X)
        nr_output_neurons = len(Y[0])
        indices = np.arange(nr_examples)
        loss_per_epoch = np.zeros(nr_output_neurons)
        for i in range(epochs):
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            for x, y in zip(X_shuffled, Y_shuffled):
                x_normed = self._norm_fct(x)
                pred_y = self._feedforward(x_normed, update_z=True)
                self._backpropagation(pred_y, y, x_normed)
                self._update_W()
                loss_per_epoch += self._loss(pred_y, np.array(y))
            print(f"loss in epoch {i+1}: ", loss_per_epoch/nr_examples)
            loss_per_epoch = np.zeros(nr_output_neurons)

    def predict(self, input):
        """
        """
        return self._feedforward(self._norm_fct(input))

    def evaluate(self, input_test, output_test):
        """
        """
        nr_fails = 0
        fail = []
        sum_loss = np.zeros(len(output_test[0]))
        nr_examples = len(output_test)
        for x, y in zip(input_test, output_test):
            pred_y = self._feedforward(self._norm_fct(x))
            sum_loss += self._loss(np.array(pred_y), np.array(y))
            if Evaluate.linear(pred_y, y):
               nr_fails = nr_fails + 1
               fail.append(x)
        accuracy = (nr_examples - nr_fails) / nr_examples
        sum_loss /= nr_examples
        return {
            "loss": sum_loss,
            "accuracy": accuracy,
            "failed_list": fail
        }

    class _FirstModelLayer:
        def __init__(self, x):
            self.z = x
            self.activation = map_activations['linear'].activation
            self.activation_der = map_activations['linear'].activation_der
            self.delta = np.empty(len(x))

    class _ModelLayer:
        def __init__(self, no_inputs: int, nr_neurons: int, activation: str | None, kernel_initializer: Callable | None, name: str | None):
            self.activation = Model.get_with_warning(map_activations, activation, map_activations['linear'], ACTIVATION_NOT_FOUND).activation
            self.activation_der = Model.get(map_activations, activation,  map_activations['linear']).activation_der
            self.W = Model.get(map_activations, self.activation,  map_activations['linear']).kernel_initializer(no_inputs, nr_neurons) if kernel_initializer is None else kernel_initializer(no_inputs, nr_neurons)
            self.der_W = np.empty(self.W.shape)
            #self.b = np.random.normal((nr_neurons, 1)) if self.activation == Activation.relu else np.zeros((nr_neurons, 1))
            self.b =  np.zeros(nr_neurons)
            self.der_b = np.zeros(nr_neurons)
            self.z = np.empty(nr_neurons)
            self.delta = np.empty(nr_neurons)
            self.name = name
