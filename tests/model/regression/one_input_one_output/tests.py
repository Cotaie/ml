import unittest
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neural.layers import Layer
from neural.model import Model
from neural.normalizations import Normalization
from neural.initializers import Initializers
from neural.optimizers import SGD as SGDD, AdaGrad
from neural.initializers import Initializers

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import SGD
from keras.initializers import RandomNormal, HeNormal

class TestModel(unittest.TestCase):

    def test_1(self):
        #               /
        #              /
        #             /
        #            /
        #-----------/
        #
        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_1_1.csv')
        X = [[item] for item in data['x'].values]
        Y = [[item] for item in data['y'].values]
        plt.scatter(X, Y, color='gray', label='input')

        x1 = np.linspace(-50,0,500)
        y1 = 0*x1 + 50
        x2 = np.linspace(0,50,500)
        y2 = 3*x2 + 50
        plt.plot(x1, y1, color='green', label='true function')
        plt.plot(x2, y2, color='green')

        mod = Model([1, Layer(1, activation="relu", kernel_initializer=Initializers.he), Layer(1, activation="linear", kernel_initializer=Initializers.random_normal)])
        mod.compile(loss='mean_squared_error', input_normalization=Normalization.z_score, clip_value=None, optimizer=SGDD(learning_rate=0.001, momentum=0.0, nesterov=False))
        mod.fit(X, Y, batch_size=100, epochs=30)
        plt.plot(X, [mod.predict(x) for x in X], color='red', label='estimated function')
        #plt.plot(X, np.squeeze([mod.predict([x]) for x in X]), color='red', label='estimated function')

        # print("_learning_rate", mod._optimizer._learning_rate)
        # print("_momentum", mod._optimizer._momentum)
        # print("_nesterov", mod._optimizer._nesterov)
        # print("_velocity", mod._optimizer._velocity)
        plt.legend()
        plt.show()

        # for e in mod._layers:
        #     print("z shape: ", np.shape(e.z))

    def test_2(self):
        #               /
        #              /
        #             /
        #            /
        #-----------/
        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_1_2.csv')
        X = [[item] for item in data['x'].values]
        Y = [[item] for item in data['y'].values]
        plt.scatter(X, Y, color='gray', label='input')

        x1 = np.linspace(-50,0,500)
        y1 = 0*x1
        x2 = np.linspace(0,50,500)
        y2 = 3*x2
        plt.plot(x1, y1, color='green', label='true function')
        plt.plot(x2, y2, color='green')

        mod = Model([1, Layer(1, activation="relu", kernel_initializer=Initializers.he)])
        mod.compile(loss='mean_squared_error', input_normalization=Normalization.z_score, clip_value=None, optimizer=SGDD(learning_rate=0.01, momentum=0.7, nesterov=False))
        mod.fit(X, Y, batch_size=1000, epochs=100)
        plt.plot(X, [mod.predict(x) for x in X], color='red', label='estimated function')
        #plt.plot(X, np.squeeze([mod.predict([x]) for x in X]), color='red', label='estimated function')

        plt.legend()
        plt.show()

    def test_3(self):
        #               /-----------
        #              /
        #             /
        #            /
        #-----------/
        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_1_3.csv')
        X = [[item] for item in data['x'].values]
        Y = [[item] for item in data['y'].values]
        plt.scatter(X, Y, color='gray', label='input')

        x1 = np.linspace(-50,0,500)
        y1 = 0*x1
        x2 = np.linspace(0,50,500)
        y2 = 3*x2
        x3 = np.linspace(50,100,500)
        y3 = 0*x3 + 150
        plt.plot(x1, y1, color='green', label='true function')
        plt.plot(x2, y2, color='green')
        plt.plot(x3, y3, color='green')

        # mod = Model([1, Layer(2, activation="relu", kernel_initializer=Initializers.he), Layer(1, activation="linear", kernel_initializer=Initializers.random_normal)])
        # mod.compile(loss='mean_squared_error', input_normalization=Normalization.z_score, optimizer=SGDD(learning_rate=0.001, momentum=0.9, nesterov=False))
        mod = Model([1, Layer(2, activation="relu", kernel_initializer=Initializers.he), Layer(2, activation="relu", kernel_initializer=Initializers.he), Layer(1, activation="linear", kernel_initializer=Initializers.random_normal)])
        mod.compile(loss='mean_squared_error', input_normalization=Normalization.z_score, clip_value=None, optimizer=SGDD(learning_rate=0.0001, momentum=0.0, nesterov=False))

        mod.fit(X, Y, batch_size=1, epochs=100)
        plt.plot(X, [mod.predict(x) for x in X], color='red', label='estimated function')
        #plt.plot(X, np.squeeze([mod.predict([x]) for x in X]), color='red', label='estimated function')

        plt.legend()
        plt.show()

    def test_4(self):
        #               /-----------\
        #              /             \
        #             /               \
        #            /                 \
        #-----------/                   \-----------
        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_1_4.csv')
        X = [[item] for item in data['x'].values]
        Y = [[item] for item in data['y'].values]
        plt.scatter(X, Y, color='gray', label='input')

        x1 = np.linspace(-50,0,500)
        y1 = 0*x1
        x2 = np.linspace(0,50,500)
        y2 = 3*x2
        x3 = np.linspace(50,100,500)
        y3 = 0*x3 + 150
        plt.plot(x1, y1, color='green', label='true function')
        plt.plot(x2, y2, color='green')
        plt.plot(x3, y3, color='green')

        mod = Model([1, Layer(5, activation="relu", kernel_initializer=Initializers.he), Layer(5, activation="relu", kernel_initializer=Initializers.he), Layer(1, activation="linear", kernel_initializer=Initializers.random_normal)])
        mod.compile(loss='mean_squared_error', input_normalization=Normalization.z_score, clip_value=None, optimizer=SGDD(learning_rate=0.00001, momentum=0.9, nesterov=False))
        mod.fit(X, Y, batch_size=100, epochs=100)
        plt.plot(X, [mod.predict(x) for x in X], color='red', label='estimated function')
        #plt.plot(X, np.squeeze([mod.predict([x]) for x in X]), color='red', label='estimated function')

        plt.legend()
        plt.show()

    def test_5(self):
        #               /-----------
        #              /
        #             /
        #            /
        #-----------/
        #
        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_1_5.csv')
        X = [[item] for item in data['x'].values]
        Y = [[item] for item in data['y'].values]
        plt.scatter(X, Y, color='gray', label='input')

        x1 = np.linspace(-50,0,500)
        y1 = 0*x1 + 50
        x2 = np.linspace(0,50,500)
        y2 = 3*x2 + 50
        x3 = np.linspace(50,100,500)
        y3 = 0*x3 + 200
        plt.plot(x1, y1, color='green', label='true function')
        plt.plot(x2, y2, color='green')
        plt.plot(x3, y3, color='green')

        # mod = Model([1, Layer(10, activation="relu", kernel_initializer=Initializers.he), Layer(10, activation="relu", kernel_initializer=Initializers.he), Layer(1, activation="linear", kernel_initializer=Initializers.random_normal)])
        # mod.compile(loss='mean_squared_error', input_normalization=Normalization.z_score, optimizer=SGDD(learning_rate=0.0001, momentum=0.9, nesterov=False))
        mod = Model([1, Layer(4, activation="relu", kernel_initializer=Initializers.he), Layer(4, activation="relu", kernel_initializer=Initializers.he), Layer(1, activation="linear", kernel_initializer=Initializers.random_normal)])
        mod.compile(loss='mean_squared_error', input_normalization=Normalization.z_score, clip_value=None, optimizer=SGDD(learning_rate=0.00002, momentum=0.9, nesterov=False))
        for i, layer in enumerate(mod._layers):
            print(f"layer {i+1} size of W: ", np.shape(layer.W))
            print(f"layer {i+1} size of der_W: ", np.shape(layer.der_W))
            print(f"layer {i+1} size of b: ", np.shape(layer.b))
            print(f"layer {i+1} size of der_b: ", np.shape(layer.der_b))
        mod.fit(X, Y, batch_size=1, epochs=100)

        plt.plot(X, [mod.predict(x) for x in X], color='red', label='estimated function')
        #plt.plot(X, np.squeeze([mod.predict([x]) for x in X]), color='red', label='estimated function')

        plt.legend()
        plt.show()

    def test_6(self):
        #\        /
        # \      /
        #  \    /
        #   \  /
        #    \/
        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_1_6.csv')
        X = [[item] for item in data['x'].values]
        Y = [[item] for item in data['y'].values]
        plt.scatter(X, Y, color='gray', label='input')

        x1 = np.linspace(-50,0,500)
        y1 = (-3)*x1
        x2 = np.linspace(0,50,500)
        y2 = 3*x2
        plt.plot(x1, y1, color='green', label='true function')
        plt.plot(x2, y2, color='green')

        mod = Model([1, Layer(2, activation="relu", kernel_initializer=Initializers.he), Layer(1, activation="linear", kernel_initializer=Initializers.random_normal)])
        mod.compile(loss='mean_squared_error', input_normalization=Normalization.z_score, clip_value=None, optimizer=SGDD(learning_rate=0.00001, momentum=0.9, nesterov=False))
        mod.fit(X, Y, batch_size=1, epochs=50)
        plt.plot(X, [mod.predict(x) for x in X], color='red', label='estimated function')
        #plt.plot(X, np.squeeze([mod.predict([x]) for x in X]), color='red', label='estimated function')

        plt.legend()
        plt.show()

    def test_keras(self):

        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_1_2.csv')
        X = [[item] for item in data['x'].values]
        Y = [[item] for item in data['y'].values]
        plt.scatter(X, Y, label='input')


        model = Sequential([Dense(units=1, activation='relu', input_shape=(1,), kernel_initializer='he_normal'), Dense(units=1, activation='linear', kernel_initializer='he_normal')])
        model.compile(loss='mse', optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=False))
        # Display the model
        model.summary()

        model.fit(X, Y, batch_size=1, epochs=20, verbose=1)

        y_predicted = model.predict(X)

        # Display the result
        plt.scatter(X[::1], Y[::1])
        plt.plot(X, y_predicted, 'r', linewidth=4)
        plt.grid()
        plt.show()
if __name__ == "__main__":
    unittest.main()