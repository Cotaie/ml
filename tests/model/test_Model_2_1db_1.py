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

class TestModel(unittest.TestCase):
    def test_two_inputs_one_output(self):
        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data\data_100_2_1db_1.csv')
        # data_validation = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\data_validation.csv')
        m1, b1 = 0, 5
        x1_values = data['x1'].values
        x2_values = data['x2'].values
        label = data['y'].values
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

        mod = Model([2, Layer(1, activation="sigmoid", kernel_initializer=None)])
        mod.compile(loss='binary_crossentropy', input_normalization=None, optimizer=SGD(learning_rate=0.01))
        mod.fit(X, Y, batch_size=5, epochs=20)

        evaluate_training = mod.evaluate(X,Y)
        print("evaluate input: ")
        print("loss: ", evaluate_training['loss'])
        print("accuracy: ", evaluate_training['accuracy'])

        evaluate_validation =  mod.evaluate(X_validation, Y_validation)
        print("evaluate validation: ")
        print("loss: ", evaluate_validation['loss'])
        print("accuracy: ", evaluate_validation['accuracy'])

        plt.plot([0, 10], [b1, 10 * m1 + b1], '-r')
        plt.scatter(x1_values[label == 0], x2_values[label == 0], label='Class 0', alpha=0.5)
        plt.scatter(x1_values[label == 1], x2_values[label == 1], label='Class 1', alpha=0.5)
        for ex in evaluate_training['failed_list']:
            plt.scatter(ex[0], ex[1], color='red')

        for ex in evaluate_validation['failed_list']:
            plt.scatter(ex[0], ex[1], color='yellow')

        plt.legend()
        plt.show()
if __name__ == "__main__":
    unittest.main()