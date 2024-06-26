import unittest
import numpy as np
from neural.layers import Layer
from neural.model import Model
from neural.initializers import Initializers
from neural.normalizations import Normalization


class TestModel(unittest.TestCase):
    def test_feedforward(self):
        X = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]]
        mlp = Model([3, Layer(3, activation="linear", kernel_initializer=Initializers.ones), Layer(2, activation="linear", kernel_initializer=Initializers.ones)], 42)
        mlp.compile(loss='mse', input_normalization=None)
        mlp._feedforward((X[0]), update_z=True)
        print("layer1 Z: ", mlp._layers[0].z)
        print("layer2 Z: ", mlp._layers[1].z)
    def test_fit(self):
        mlp = Model([2, Layer(3, activation="linear"), Layer(2, activation="linear")], 42)
        mlp.compile(loss="mse", input_normalization=None)
        if not np.allclose(mlp._layers[0].W, np.array([[ 0., 0.00496714, -0.00138264], [0., 0.00647689, 0.0152303], [0., -0.00234153, -0.00234137]])):
            self.fail("Layer 0 weights do not match expectations")
        if not np.allclose(mlp._layers[1].W, np.array([[ 0., 0.01579213, 0.00767435, -0.00469474], [0., 0.0054256, -0.00463418, -0.0046573]])):
            self.fail("Layer 1 weights do not match expectations")
        if not np.allclose(mlp.predict([2, 3]), np.array([ 0.000596400577011, -0.000185851512241])):
            print(mlp.predict([2, 3]))
            self.fail("Prediction fail")
        mlp.fit([[2, 3]], [[1, 2]])
        if not np.allclose(mlp._layers[1].der_W, np.array([[-1.99880719884598, -0.011565806030271, -0.117219381774002, 0.023400388066718], [-4.00037170302448, -0.023147566805283, -0.234600464799945, 0.046833056392802]])):
            print(mlp._layers[1].der_W)
            self.fail("Layer 1 der W fail")
        if not np.allclose(mlp._layers[0].der_W, np.array([[-0.053269837843337, -0.106539675686674, -0.159809513530011], [0.003198889507886, 0.006397779015771, 0.009596668523657], [0.028014809035625, 0.05602961807125, 0.084044427106875]])):
            self.fail("Layer 0 der W fail")

if __name__ == "__main__":
    unittest.main()
