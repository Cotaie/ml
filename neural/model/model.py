import numpy as np
import warnings
from itertools import chain
from typing import Callable
from neural.normalizations import Normalization
from neural.optimizers import SGD
from neural.evaluations import Evaluate
from neural.model.maps import MapActivation, MapLoss, map_loss, map_activations
from neural._basic import BasicLayer
from neural.constants import BIAS_INPUT, LOSS_NOT_FOUND, ACTIVATION_NOT_FOUND


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
    def _clip_W(weights):
        """
        """
        np.clip(weights, a_min=-5, a_max=5, out=weights)
    @staticmethod
    def _normalize_der_W(der_W, norm_value=1.0):
        """
        """
        norm = np.linalg.norm(der_W)
        #print("norm: ", norm)
        if norm == 0:
            return der_W
        else:
            return (der_W / norm) * norm_value

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
        current_output = input[:]
        def ff_update_z(layer):
            nonlocal current_output
            input_with_bias = np.concatenate(([BIAS_INPUT], current_output))
            z = layer.W @ input_with_bias
            layer.z[:] = z
            current_output = layer.activation(z)
        def ff_no_update_z(layer):
            nonlocal current_output
            input_with_bias = np.concatenate(([BIAS_INPUT], current_output))
            z = layer.W @ input_with_bias
            current_output = layer.activation(z)
        ff = ff_update_z if update_z else ff_no_update_z
        for layer in self._layers:
            ff(layer)
        return current_output

    def _update_W(self):
        """
        """
        for layer in self._layers:
            self._optimizer.step(layer)

    def _backpropagation(self, output, x):
        """
        """
        delta_layer = output[:]
        layers_reversed = chain(reversed(self._layers), iter([Model._FirstModelLayer(x)]))
        prev_layer = next(layers_reversed)
        for layer in layers_reversed:
            prev_layer.der_W[:] = [delta * layer.activation(np.concatenate(([BIAS_INPUT], layer.z))) for delta in delta_layer]
            if (True):
                #Model._clip_W(prev_layer.der_W)
                prev_layer.der_W = Model._normalize_der_W(prev_layer.der_W)
            delta_layer = delta_layer @ prev_layer.W[:, 1:]
            prev_layer = layer

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
        indices = np.arange(len(X))
        loss_per_epoch = np.zeros(len(Y[0]))
        for i in range(epochs):
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            for x, y in zip(X_shuffled, Y_shuffled):
                x_normed = self._norm_fct(x)
                output = self._feedforward(x_normed, update_z=True)
                loss_per_epoch += self._loss(output, np.array(y))
                self._backpropagation(self._loss_der(output, np.array(y)), x_normed)
                self._update_W()
            print(f"loss in epoch {i+1}: ", loss_per_epoch/len(Y))
            loss_per_epoch = np.zeros(len(Y[0]))

    # def fit(self, X, Y, batch_size, epochs):
    #     #self._norm_fct = self._input_normalization(X)
    #     for _ in range(epochs):
    #         get_batch = Model._batch(X, Y, batch_size)
    #         for X_batch, Y_batch in get_batch:
    #             batch_loss_der = np.array([0.])
    #             for x, y in zip(X_batch, Y_batch):
    #                 x_normed = self._norm_fct(x)
    #                 #np.add(batch_loss_der, np.array(self._comp_loss_der_arr(self._feed_forward(x_normed, True), y)), out=batch_loss_der)
    #                 batch_loss_der += np.array(self._comp_loss_der_arr(self._feed_forward(x_normed, True), y))
    #             len_mse = len(Y_batch)
    #             #print("batch_size: ", len_mse)
    #             #print("abg_batch:", batch_loss_der)
    #             batch_loss_der /= float(len_mse)
    #             for x, y in zip(X_batch, Y_batch):
    #                 x_normed = self._norm_fct(x)
    #                 self._compute_gradients(batch_loss_der, x_normed)
    #                 self._adjust_W()

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

    class _ModelLayer:
        def __init__(self, no_inputs: int, nr_neurons: int, activation: str | None, kernel_initializer: Callable | None, name: str | None):
            self.activation = Model.get_with_warning(map_activations, activation, map_activations['linear'], ACTIVATION_NOT_FOUND).activation
            self.activation_der = Model.get(map_activations, activation,  map_activations['linear']).activation_der
            self.W = Model.get(map_activations, self.activation,  map_activations['linear']).kernel_initializer(no_inputs, nr_neurons) if kernel_initializer is None else kernel_initializer(no_inputs, nr_neurons)
            self.der_W = np.empty(self.W.shape)
            self.z = np.empty(nr_neurons)
            self.name = name
