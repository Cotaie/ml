import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural import Layer, Model
from constants import SIGMOID_MIDPOINT

def find_y(target_x1, target_x2, df):
    # Search for the rows that match the target x1 and x2 values
    matching_rows = df[(df['x1'] == target_x1) & (df['x2'] == target_x2)]

    if not matching_rows.empty:
        return matching_rows['y'].values[0]
    else:
        return None

class TestModel(unittest.TestCase):
    def test_two_inputs_one_output(self):
        data = pd.read_csv('data_fin_3.csv')
        m1, b1 = -1, 10
        x1_values = data['x1'].values
        x2_values = data['x2'].values
        label = data['y'].values
        X = [list(item) for item in zip(data['x1'].values, data['x2'].values)]
        Y = [[item] for item in data['y'].values]
        mod = Model([2, Layer(1, activation="sigmoid")])
        mod.compile(loss='binary_crossentropy')
        # mean = np.mean(X, axis=0)
        # std = np.std(X, axis=0)
        # normalized_X = (X - mean) / std
        # mod.fit(normalized_X, Y, batch_size=1,epochs=10)
        mod.fit(X, Y, batch_size=1,epochs=10)
        no_fails = 0
        fail = []
        for index,x in enumerate(X):
            if (0 if mod.predict(x) < SIGMOID_MIDPOINT else 1) != data.iloc[index]['y']:
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