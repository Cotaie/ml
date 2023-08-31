import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural import Layer, Model
from normalization import Normalization
from initializers import Initializers


class TestModel(unittest.TestCase):
    def test_two_inputs_one_output(self):
        data = pd.read_csv('data_bin_m1_10.csv')
        m1, b1 = -1, 10
        x1_values = data['x1'].values
        x2_values = data['x2'].values
        label = data['y'].values
        X = [list(item) for item in zip(data['x1'].values, data['x2'].values)]
        Y = [[item] for item in data['y'].values]
        mod = Model([2, Layer(1, activation="sigmoid", kernel_initializer=Initializers.xavier_normal)])
        mod.compile(loss='binary_crossentropy', input_normalization=Normalization.z_score)
        mod.fit(X, Y, epochs=40)
        no_fails = 0
        fail = []
        for index,x in enumerate(X):
            if mod.predict(x) != data.iloc[index]['y']:
                no_fails = no_fails+1
                fail.append(x)
                #print(f"fail: {x}, ", f"csv: {data.iloc[index]['y']}", f"predicted {mod.predict(x)}")
        for ex in fail:
            plt.scatter(ex[0], ex[1], color='black')
        print("number of fails:", no_fails)
        plt.plot([0, 10], [b1, 10 * m1 + b1], '-r')
        plt.scatter(x1_values[label == 0], x2_values[label == 0], label='Class 0', alpha=0.5)
        plt.scatter(x1_values[label == 1], x2_values[label == 1], label='Class 1', alpha=0.5)
        plt.legend()
        plt.show()
if __name__ == "__main__":
    unittest.main()