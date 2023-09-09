import numpy as np
import warnings
from itertools import chain
from typing import Callable
from constants import BIAS_INPUT, LOSS_NOT_FOUND, ACTIVATION_NOT_FOUND
from normalization import Normalization
from optimizers import SGD
from evaluations import Evaluate
from maps import MapActivation, MapLoss, map_loss, map_activations


class _BasicLayer:
    def __init__(self, units: int):
        self.units = units

class Layer(_BasicLayer):
    def __init__(self, units: int, activation: str | None = None, kernel_initializer: Callable | None = None, name: str | None = None):
        super().__init__(units)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.name = name

    def get_name_or_default(self, layer_no: int) -> str:
        """
        Retrieves the name of the layer if it's set, otherwise provides a default name based on the layer number.
        Parameters:
        - layer_no (int): The sequence number of the layer, used to generate a default name if the layer's name is not set.
        Returns:
        - str: The name of the layer or a default name in the format "layer_{layer_no}".
        """
        return self.name if self.name is not None else f"layer_{layer_no}"

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
        Retrieves the value associated with a given key from a dictionary.
        If the key is not present, it issues a warning and returns a default value.
        Parameters:
        - dict (dict): The dictionary from which to retrieve the value.
        - key (str): The key whose value needs to be retrieved.
        - default (Callable): A default value or callable to return if the key is not present.
        - warning (str): The warning message to issue if the key is not present.
        Returns:
        - The value associated with the key if present, otherwise the default value.
        Warnings:
        - Issues a warning if the key is not present in the dictionary.
        """
        if key not in dict:
            warnings.warn(warning)
        return dict.get(key, default)
    @staticmethod
    def get(dict: dict, key: str, default: MapActivation | MapLoss) -> Callable:
          return dict.get(key, default)
    @staticmethod
    def _batch(input, output, batch_size):
        """
        Generate batches of data from the provided input and output lists (or arrays).
        Parameters:
        - input (list or array-like): The input data list or array.
        - output (list or array-like): The corresponding output (labels or target values) list or array.
        - batch_size (int): The size of each batch.
        Yields:
        - tuple: A tuple containing a batch of input data and its corresponding output. The size of each
                batch is determined by `batch_size`, except possibly for the last batch which might be smaller.
        """
        num_batches = (len(input) + batch_size - 1) // batch_size
        for i in range(num_batches):
            yield (input[i * batch_size:(i + 1) * batch_size], output[i * batch_size:(i + 1) * batch_size])
    @staticmethod
    def _clip_W(weights):
        np.clip(weights, a_min=-5, a_max=5, out=weights)
    def _build(self, model_arch: list, seed: int | None):
        """
        Builds model layers given architecture.
        Parameters:
        - model_arch (list): List of layers where each layer is an object with attributes
                            detailing its architecture (e.g., number of units, activation
                            function, kernel initializer, etc.).
        - seed (int | None): Seed for the numpy random number generator. Setting this ensures
                            reproducible random behaviors. If None, seeding is not applied.
        Returns:
        - list: A list of model layers that is constructed based on
                the given architecture and the number of units in the previous layer.
        """
        np.random.seed(seed)
        previous_layer = _BasicLayer(model_arch[0])
        def _build_layer(layer, layer_index):
            nonlocal previous_layer
            model_layer = Model._ModelLayer(previous_layer.units, layer.units, layer.activation, layer.kernel_initializer, layer.get_name_or_default(layer_index))
            previous_layer = layer
            return model_layer
        return [_build_layer(layer, layer_index) for layer_index, layer in enumerate(model_arch[1:], start=1)]

    def _feedforward(self, input, update_z=False):
        """
        Propagate the input through the network layers.
        Parameters:
        - input (np.array): The input data to feed forward through the model.
        - update_z (bool, optional): Flag to determine if the intermediate z-values should be updated. Default is False.
        Returns:
        - current_output (np.array): The output after passing through all layers.
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
        Updates the weights of each layer in the model using the equation:
        ΔW = - learning_rate * derivative_of_W
        The weights are updated in-place.
        """
        optimizer = self._optimizer
        for layer in self._layers:
            optimizer.step(weights=layer.W, gradient=layer.der_W)

    def _backpropagation(self, output, x):
        """
        Performs the backpropagation algorithm to compute the gradient of the loss
        with respect to the network's weights for a given input and output.
        The method traverses the network in reverse (from output to input) to
        compute the gradient for each layer based on the "delta" (error gradient)
        of the subsequent layer.
        Parameters:
        - output (array-like): The derivative of the loss with respect to the
                            network's final output.
        - x (array-like): The input data sample.
        """
        delta_layer = output[:]
        layers_reversed = chain(reversed(self._layers), iter([Model._FirstModelLayer(x)]))
        prev_layer = next(layers_reversed)
        for layer in layers_reversed:
            prev_layer.der_W[:] = [delta * layer.activation(np.concatenate(([BIAS_INPUT], layer.z))) for delta in delta_layer]
            if (False):
                Model._clip_W(prev_layer.der_W)
            delta_layer = delta_layer @ prev_layer.W[:, 1:]
            prev_layer = layer

    def compile(self, optimizer=SGD(), loss=None, input_normalization=None):
        """
        Compile the model by setting up the optimizer, loss function, and input normalization method.
        Parameters:
        - optimizer (optional): The optimization method to use for training the model. If not provided,
                                the model will retain its previous optimizer or have no optimizer if it
                                hasn't been set before.
        - loss (str or None, optional): The name/key of the loss function to be used. If not provided,
                                    the 'mean_squared_error' loss function is set as default. If the given
                                    loss name/key is not found in the `map_loss` dictionary, a warning
                                    will be issued and 'mean_squared_error' will be used.
        - input_normalization (callable or None, optional): The normalization method to apply to the
                                                        model input data. If not provided,
                                                        'no_normalization' is set as default.
        """
        self._optimizer = optimizer
        self._loss = Model.get_with_warning(map_loss, loss, map_loss['mean_squared_error'], LOSS_NOT_FOUND).loss
        self._loss_der = Model.get(map_loss, loss, map_loss['mean_squared_error']).loss_der
        self._input_normalization = Normalization.no_normalization if input_normalization is None else input_normalization

    def fit(self, X, Y, batch_size=32, epochs=1):
        """
        Trains the neural network using the provided input data and labels.
        Parameters:
        - X (array-like): Input data for training. Each element of X is a single sample.
        - Y (array-like): Target labels corresponding to each sample in X.
        - batch_size (int, optional): Size of batches for training. Defaults to 32.
        - epochs (int, optional): Number of times the training data should be iterated over. Defaults to 1.
        """
        # X = np.array(X)
        # Y = np.array(Y)
        loss_per_epoch = np.zeros(len(Y[0]))
        for i in range(1, epochs+1):
            # indices = np.arange(X.shape[0])
            # np.random.shuffle(indices)
            # X = X[indices]
            # Y = Y[indices]
            for x, y in zip(X, Y):
                x_normed = self._norm_fct(x)
                output = self._feedforward(x_normed, update_z=True)
                loss_per_epoch += self._loss(output, np.array(y))
                self._backpropagation(self._loss_der(output, np.array(y)), x_normed)
                self._update_W()
            print(f"loss in epoch {i}: ", loss_per_epoch/len(Y))
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
        return self._feedforward(self._norm_fct(input))

    def evaluate(self, input_test, output_test):
        nr_fails = 0
        sum_loss = np.zeros(len(output_test[0]))
        nr_examples = len(output_test)
        for x, y in zip(input_test, output_test):
            pred_y = self._feedforward(self._norm_fct(x))
            sum_loss += self._loss(np.array(pred_y), np.array(y))
            if Evaluate.binary_classification(pred_y, y):
               nr_fails = nr_fails + 1
        accuracy = (nr_examples - nr_fails) / nr_examples
        sum_loss /= nr_examples
        return {
            "loss": sum_loss,
            "accuracy": accuracy
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
