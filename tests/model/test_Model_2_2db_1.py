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
from neural.split_train_validation import split_train_validation
from neural.initializers import Initializers

class TestModel(unittest.TestCase):
    def test_two_inputs_one_output(self):
        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_10000_2_2db_1.csv')
        # data_validation = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data_validation.csv')

        m1, b1 = 1, 33   # First line
        m2, b2 = -1, 66  # Second line

        X = [list(item) for item in zip(data['x1'].values, data['x2'].values)]
        Y = [[item] for item in data['y'].values]

        data = split_train_validation(X,Y, seed=14)
        X, Y = data['training']
        X_validation, Y_validation = data['validation']
        # x1_validation = data_validation['x1'].values
        # x2_validation = data_validation['x2'].values
        # label_validation = data['y'].values
        # X_validation = [list(item) for item in zip(data_validation['x1'].values, data_validation['x2'].values)]
        # Y_validation = [[item] for item in data_validation['y'].values]

        mod = Model([2, Layer(2, activation='leaky_relu', kernel_initializer=Initializers.he), Layer(1, activation="sigmoid", kernel_initializer=Initializers.he)])
        mod.compile(loss='binary_crossentropy', input_normalization=Normalization.min_max, optimizer=SGD(learning_rate=0.00005, momentum=0.8))
        mod.fit(X, Y, batch_size=5, epochs=20)

        print("W 1: ", mod._layers[0].W)
        print("der W 1: ", mod._layers[0].der_W)

        print("W 2: ", mod._layers[1].W)
        print("der W 2: ", mod._layers[1].der_W)

        evaluate_training = mod.evaluate(X,Y)
        print("evaluate input: ")
        print("loss: ", evaluate_training['loss'])
        print("accuracy: ", evaluate_training['accuracy'])

        evaluate_validation =  mod.evaluate(X_validation, Y_validation)
        print("evaluate validation: ")
        print("loss: ", evaluate_validation['loss'])
        print("accuracy: ", evaluate_validation['accuracy'])

        plt.plot([0, 100], [b1, 100 * m1 + b1], '-r')
        plt.plot([0, 100], [b2, 100 * m2 + b2], '-r')
        #plt.scatter(np.array(X)[:,0], np.array(X)[:,1],c=[item for sublist in Y for item in sublist])
        plt.scatter(np.array(X)[:,0], np.array(X)[:,1])
        #print("test: ", X[:,1])

        for ex in evaluate_training['failed_list']:
            plt.scatter(ex[0], ex[1], color='red')

        for ex in evaluate_validation['failed_list']:
            plt.scatter(ex[0], ex[1], color='yellow')

        plt.legend()
        plt.show()
if __name__ == "__main__":
    unittest.main()