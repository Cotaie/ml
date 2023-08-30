import unittest
import numpy as np
import pandas as pd
from neural import Layer, Model

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
        #print(data)
        X = [list(item) for item in zip(data['x1'].values, data['x2'].values)]
        Y = [[item] for item in data['y'].values]
        mod = Model([2, Layer(1, activation="sigmoid")])
        mod.compile(loss='binary_crossentropy')
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        normalized_X = (X - mean) / std
        mod.fit(normalized_X, Y, epochs=10)
        no_fails = 0
        for index,x in enumerate(X):
            if mod.predict((np.array(x) - mean) / std) != data.iloc[index]['y']:
                no_fails = no_fails+1
        print("No of fails: ", no_fails)

if __name__ == "__main__":
    unittest.main()