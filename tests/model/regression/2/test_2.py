import unittest
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neural.layers import Layer
from neural.model import Model
from neural.normalizations import Normalization
from neural.initializers import Initializers
from neural.optimizers import SGD
from neural.initializers import Initializers

class TestModel(unittest.TestCase):
    def test_one_input_one_output(self):
        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_1_1.csv')
        X = [[item] for item in data['x'].values]
        Y = [[item] for item in data['y'].values]
        plt.scatter(X, Y, label='input')

        a, b, c = 0, 3, 2
        x = np.linspace(-50, 50, 1000)
        y = a * x**2 + b * abs(x) + c
        plt.plot(x, y, color='green', label='true function')

        mod = Model([1, Layer(2, activation="leaky_relu", kernel_initializer=Initializers.he), Layer(1, activation="linear", kernel_initializer=Initializers.he)])
        mod.compile(loss='mean_squared_error', input_normalization=Normalization.no_normalization, optimizer=SGD(learning_rate=0.001, momentum=0.7, nesterov=False))
        mod.fit(X, Y, batch_size=5, epochs=100)
        plt.plot(X, [mod.predict(x) for x in X], 'red', label='estimated function')

        plt.legend()
        plt.show()
if __name__ == "__main__":
    unittest.main()