import numpy as np
import unittest
from neural import Layer, Model
from activations import ident, sigmoid, ident_der, sigmoid_der

class TestModel(unittest.TestCase):
    def test_Layer1(self):
        mlp_arch = [2, Layer(3, activation='linear'), Layer(2, activation='linear')]
        mlp = Model(mlp_arch)
        mlp.compile(loss='mse')
        mlp._set_W_1()
        self.assertTrue((mlp._model[0]._W == np.array([[1,1,1],[1,1,1],[1,1,1]])).all())
        self.assertTrue((mlp._model[1]._W == np.array([[1,1,1,1],[1,1,1,1]])).all())

if __name__ == '__main__':
    unittest.main()