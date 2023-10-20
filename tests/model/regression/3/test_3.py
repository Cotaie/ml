import unittest
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neural.layers import Layer
from neural.model import Model
from neural.normalizations import Normalization
from neural.initializers import Initializers
from neural.optimizers import SGD as SSGD
from neural.initializers import Initializers
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import SGD

class TestModel(unittest.TestCase):
    def test_one_input_one_output(self):
        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_1_4.csv')
        X = [[item] for item in data['x'].values]
        Y = [[item] for item in data['y'].values]
        plt.scatter(X, Y, color='blue', label='input data')

        s = 3
        x1 = np.linspace(-50,0,500)
        y1 = 0*x1
        x2 = np.linspace(0, 50, 500)
        y2 = s * x2
        x3 = np.linspace(50,100,500)
        y3 = 0*x1 + 150
        x4 = np.linspace(100,150,500)
        y4 = -s * x4 + 450
        x5 = np.linspace(150,200,500)
        y5 = 0*x1
        plt.plot(x1, y1, color='green', label='real function')
        plt.plot(x2, y2, color='green')
        plt.plot(x3, y3, color='green')
        plt.plot(x4, y4, color='green')
        plt.plot(x5, y5, color='green')

        mod = Model([1, Layer(1, activation="relu", kernel_initializer=Initializers.xavier_normal), Layer(1, activation="relu", kernel_initializer=Initializers.xavier_normal), Layer(1, activation="relu", kernel_initializer=Initializers.xavier_normal), Layer(1, activation="relu", kernel_initializer=Initializers.xavier_normal)])
        mod.compile(loss='mean_squared_error', input_normalization=Normalization.no_normalization, optimizer=SSGD(learning_rate=0.01, momentum=0.8, nesterov=False))
        mod.fit(X, Y, batch_size=5, epochs=50)
        plt.plot(X, [mod.predict(x) for x in X], color='red', label='estimated function')
        # for _ in range(50):
        #     mod.fit(X, Y, batch_size=5, epochs=1)
        #     plt.scatter(X, Y, color='blue', label='input data')
        #     plt.plot(x1, y1, color='green', label='real function')
        #     plt.plot(x2, y2, color='green')
        #     plt.plot(x3, y3, color='green')
        #     plt.plot(x4, y4, color='green')
        #     plt.plot(x5, y5, color='green')
        #     plt.plot(X, [mod.predict(x) for x in X], color='red', label='estimated function')
        #     plt.pause(0.1)
        #     plt.clf()

        #print("w:", mod._layers[0].der_W)
        plt.legend()
        plt.show()

    # def test_keras(self):
    #     data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_1_1.csv')
    #     X = [item for item in data['x'].values]
    #     Y = [item for item in data['y'].values]
    #     plt.scatter(X, Y, label='input')
    #     model = Sequential([
    #         Dense(4,input_dim=1,activation='relu', kernel_initializer='he_normal', name='layer1'),
    #         BatchNormalization(),
    #         Dense(1, activation='linear', kernel_initializer='he_normal', name='layer2')])
    #     model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='mean_squared_error')
    #     model.fit(X, Y, epochs=10, batch_size=10)
    #     #plt.plot(X, [model.predict([x], verbose=0)[0] for x in X], 'red', label='estimated function')
    #     print("bla: ", model.predict([X[0]], verbose=1)[0])
    #     # print("bla: ", np.array([X[0]]).shape)
    #     # print("bla: ", model.predict([X[0]]).shape)
    #     #model.summary()
    #     plt.show()

if __name__ == "__main__":
    unittest.main()