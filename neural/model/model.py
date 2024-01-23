import numpy as np
from numpy.typing import NDArray
from typing import Callable
from neural.normalizations import Normalization
from neural.optimizers import SGD, AdaGrad
from neural.evaluations import Evaluate
from neural.model.maps import map_loss, map_activations
from neural._basic import BasicLayer


class Model:
    def __init__(self, model_arch: list, seed: int|None = None):
        self._layers = self._build(model_arch, seed)
        self._norm_fct = Normalization.no_normalization(None)
    @staticmethod
    def _batch(input, output, batch_size):
        """
        """
        num_batches = (len(input) + batch_size - 1) // batch_size
        for i in range(num_batches):
            yield (input[i * batch_size:(i + 1) * batch_size], output[i * batch_size:(i + 1) * batch_size])

    def _build(self, model_arch: list, seed: int|None):
        """
        """
        np.random.seed(seed)
        previous_layer = BasicLayer(model_arch[0])
        def _build_layer(layer, layer_index):
            nonlocal previous_layer
            model_layer = Model._ModelLayer(nr_inputs=previous_layer.units, nr_neurons=layer.units, activation=layer.activation, kernel_initializer=layer.kernel_initializer, name=layer.get_name_or_default(layer_index))
            previous_layer = layer
            return model_layer
        return [_build_layer(layer, layer_index) for layer_index, layer in enumerate(model_arch[1:], start=1)]

    def _feedforward(self, input: NDArray[np.number], update_z=False):
        """
        act (k, n_prev): previous activation
        W (n_prev, n): weight matrix
        b (n,): bias vector
        z (k, n): pre-activation
            k = mini-batch size,
            n_prev = number of previous neurons,
            n = number of neurons
        """
        act = input[:]
        for layer in self._layers:
            z = act@layer.W  + layer.b
            if update_z:
                layer.z[:] = z
            act = layer.activation(z)
        return act

    def _backpropagation(self,pred_y, y, x):
        """
        """
        layers_reversed = list(reversed(self._layers))
        prev_layer = layers_reversed[0]
        # 
        prev_layer.delta[:] = self._loss_der(pred_y, np.array(y)) * prev_layer.activation_der(prev_layer.z)
        #print("first delta shape: ", np.shape(prev_layer.delta))
        for layer in layers_reversed[1:]:
            prev_layer.der_b[:] = np.mean(prev_layer.delta, axis=0)
            prev_layer.der_W[:] = np.array([np.mean(prev_layer.delta * layer.activation(layer.z), axis=0)]).T
            layer.delta[:] = (prev_layer.delta@prev_layer.W.T) * layer.activation_der(layer.z)
            prev_layer = layer
        #print("spanac")
        layers_reversed[-1].der_b[:] =  np.mean(prev_layer.delta, axis=0)
        layers_reversed[-1].der_W[:] = np.mean(prev_layer.delta * x, axis=0)

    def _clip_der(self):
        """
        """
        for layer in self._layers:
            np.clip(layer.der_b, a_min=-self._clip_value, a_max=self._clip_value, out=layer.der_b)
            np.clip(layer.der_W, a_min=-self._clip_value, a_max=self._clip_value, out=layer.der_W)

    def compile(self, optimizer: SGD|AdaGrad=SGD(), loss='mse', input_normalization=Normalization.no_normalization, clip_value=None):
        """
        """
        self._optimizer = optimizer
        self._loss = map_loss[loss].loss
        self._loss_der = map_loss[loss].loss_der
        self._input_normalization = input_normalization
        self._clip_value = clip_value

    # def fit(self, X, Y, batch_size=32, epochs=1):
    #     """
    #     """
    #     self._norm_fct = self._input_normalization(X)
    #     nr_examples = len(X)
    #     nr_output_neurons = len(Y[0])
    #     loss_per_epoch = np.zeros(nr_output_neurons)
    #     for layer in self._layers:
    #         layer.z = np.zeros((batch_size, layer.b.shape[0]))
    #         layer.delta = np.zeros((batch_size, layer.b.shape[0]))
    #     for i in range(epochs):
    #         nr = 0
    #         for batch in self._batch(X, Y, batch_size):
    #             nr = nr + 1
    #             x_normed_mini_batch = self._norm_fct(batch[0])
    #             y_mini_batch = batch[1]
    #             pred_y = self._feedforward(x_normed_mini_batch, update_z=True)
    #             self._backpropagation(pred_y, y_mini_batch, x_normed_mini_batch)
    #             if self._clip_value is not None:
    #                 self._clip_der()
    #             self._optimizer.step(self._layers)
    #             loss = self._loss(pred_y, np.array(y_mini_batch))
    #             #print("loss value: ", loss)
    #             loss_per_epoch += np.sum(loss)
    #         print(f"loss in epoch {i+1}: ", loss_per_epoch/nr_examples)
    #         loss_per_epoch = np.zeros(nr_output_neurons)

    def fit(self, X, Y, batch_size=32, epochs=1):
        """
        """
        self._norm_fct = self._input_normalization(X)
        X = np.array(X)
        Y = np.array(Y)
        nr_examples = len(X)
        nr_output_neurons = len(Y[0])
        indices = np.arange(nr_examples)
        loss_per_epoch = np.zeros(nr_output_neurons)
        for layer in self._layers:
            layer.z = np.zeros((batch_size, layer.b.shape[0]))
            layer.delta = np.zeros((batch_size, layer.b.shape[0]))
        # for i, layer in enumerate(self._layers):
        #     print(f"layer {i+1} size of z: ", np.shape(layer.z))
        #     print(f"layer {i+1} size of delta: ", np.shape(layer.delta))
        for i in range(epochs):
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            for x, y in zip(X_shuffled, Y_shuffled):
                x_normed = np.array([self._norm_fct(x)])
                y = np.array([y])
                pred_y = self._feedforward(x_normed, update_z=True)
                self._backpropagation(pred_y, y, x_normed)
                if self._clip_value is not None:
                    self._clip_der()
                self._optimizer.step(self._layers)
                loss_per_epoch += np.mean(self._loss(pred_y, np.array(y)))
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

    class _ModelLayer:
        def __init__(self, name: str, nr_inputs: int, nr_neurons: int, kernel_initializer, activation='linear'):
            self.activation = map_activations[activation].activation
            self.activation_der = map_activations[activation].activation_der
            self.W = map_activations[activation].kernel_initializer(nr_inputs, nr_neurons) if kernel_initializer is None else kernel_initializer(nr_inputs, nr_neurons)
            self.der_W = np.empty(self.W.shape)
            self.b =  np.zeros(nr_neurons)
            self.der_b = np.zeros(nr_neurons)
            self.z = None
            self.delta = None
            self.name = name
