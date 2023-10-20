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
    def test_no_activation1(self):
        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_b2_s3.csv')
        X = [[item] for item in data['x'].values]
        Y = [[item] for item in data['y'].values]
        plt.scatter(X, Y, label='input')

        b, s = 2, 3
        x = np.linspace(-50, 50, 1000)
        y = b + s*x
        plt.plot(x, y, color='green', label='true function')

        mod = Model([1, Layer(1, activation="linear", kernel_initializer=Initializers.random_normal)])
        print("init: ", mod._layers[0].W)
        mod.compile(loss='mean_squared_error', input_normalization=Normalization.no_normalization, optimizer=SGD(learning_rate=0.01, momentum=0.9))
        mod.fit(X, Y, batch_size=1, epochs=50)
        plt.plot(X, [mod.predict(x) for x in X], '-r', label = 'estimated function')
        #plt.plot(X, np.squeeze([mod.predict([x]) for x in X]), '-r', label = 'estimated function')
        print("W0: ", mod._layers[0].W)
        print("velocity: ", mod._optimizer._velocity)
        plt.legend()
        plt.show()
    # def test_no_activation2(self):
    #     data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_b2_s3.csv')
    #     X = [[item] for item in data['x'].values]
    #     Y = [[item] for item in data['y'].values]
    #     plt.scatter(X, Y, label='input', color='blue')

    #     b, s = 2, 3
    #     x = np.linspace(-50, 50, 1000)
    #     y = b + s*x
    #     plt.plot(x, y, color='green', label='true function')

    #     mod = Model([1, Layer(1, activation="linear", kernel_initializer=Initializers.random_normal)])
    #     mod.compile(loss='mean_squared_error', input_normalization=Normalization.z_score, optimizer=SGD(learning_rate=0.001, momentum=0.8))

    #     #mod.fit(X, Y, batch_size=5, epochs=100)
    #     #plt.plot(X, [mod.predict(x) for x in X], '-r', label = 'estimated function')
    #     for _ in range(100):
    #         mod.fit(X, Y, batch_size=5, epochs=1)
    #         plt.plot(x, y, color='green', label='true function')
    #         plt.scatter(X, Y, color='blue', label='input')
    #         plt.plot(X, [mod.predict(x) for x in X], '-r', label='estimated function')
    #         plt.pause(0.1)
    #         plt.clf()
    #     print("W: ", mod._layers[0].W)

    # def test_relu(self):
    #     data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_b2_s3.csv')
    #     X = [[item] for item in data['x'].values]
    #     Y = [[item] for item in data['y'].values]
    #     plt.scatter(X, Y, label='input')

    #     b, s = 2, 3
    #     x = np.linspace(-50, 50, 1000)
    #     y = b + s*x
    #     plt.plot(x, y, color='green', label='true function')

    #     mod = Model([1, Layer(5, activation="leaky_relu", kernel_initializer=Initializers.he), Layer(1, activation="linear", kernel_initializer=Initializers.he)])
    #     mod.compile(loss='mean_squared_error', input_normalization=Normalization.z_score, optimizer=SGD(learning_rate=0.0005, momentum=0.9))
    #     #mod.fit(X, Y, batch_size=5, epochs=100)
    #     for _ in range(100):
    #         mod.fit(X, Y, batch_size=5, epochs=1)
    #         plt.plot(x, y, color='green', label='true function')
    #         plt.scatter(X, Y, label='input')
    #         plt.plot(X, [mod.predict(x) for x in X], '-r', label = 'estimated function')
    #         plt.pause(0.1)
    #         plt.clf()
    #         #plt.show()
    #     #plt.legend()
    #     #plt.show()
    #     print("W0: ", mod._layers[0].W[0])
    #     print("W1: ", mod._layers[0].W[1])
    #     print("W2: ", mod._layers[0].W[2])
    #     print("W3: ", mod._layers[0].W[3])
    #     print("W4: ", mod._layers[0].W[4])
if __name__ == "__main__":
    unittest.main()