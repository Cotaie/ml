import unittest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neural import Layer, Model
from normalization import Normalization
from initializers import Initializers
from optimizers import SGD
from constants import SIGMOID_MIDPOINT

class TestModel(unittest.TestCase):
    def test_two_inputs_one_output(self):
        data = pd.read_csv('data_bin_m1_10.csv')
        data_validation = pd.read_csv('data_validation.csv')
        m1, b1 = -1, 10
        x1_values = data['x1'].values
        x2_values = data['x2'].values
        label = data['y'].values
        X = [list(item) for item in zip(data['x1'].values, data['x2'].values)]
        Y = [[item] for item in data['y'].values]

        x1_validation = data_validation['x1'].values
        x2_validation = data_validation['x2'].values
        label_validation = data['y'].values
        X_validation = [list(item) for item in zip(data_validation['x1'].values, data_validation['x2'].values)]
        Y_validation = [[item] for item in data_validation['y'].values]

        mod = Model([2, Layer(1, activation="sigmoid", kernel_initializer=None)])
        mod.compile(loss='binary_crossentropy', input_normalization=None, optimizer=SGD())
        mod.fit(X, Y, batch_size=5, epochs=70)
        print("evaluate input: ", mod.evaluate(X,Y))
        print("evaluate test_input: ", mod.evaluate(X_validation,Y_validation))

        # plt.plot([0, 10], [b1, 10 * m1 + b1], '-r')
        # plt.scatter(x1_values[label == 0], x2_values[label == 0], label='Class 0', alpha=0.5)
        # plt.scatter(x1_values[label == 1], x2_values[label == 1], label='Class 1', alpha=0.5)
        # plt.legend()
        # plt.show()
if __name__ == "__main__":
    unittest.main()