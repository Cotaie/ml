import numpy as np
import unittest
from neural import Layer, Model
from activations import ident, sigmoid, ident_der, sigmoid_der


class TestModel(unittest.TestCase):
    def test_Layer1(self):
        mlp_arch = [2, Layer(3, activation="linear"), Layer(2, activation="linear")]
        mlp = Model(mlp_arch)
        mlp.compile(loss="mse")
        mlp._set_W_1()
        self.assertTrue((mlp._model[0]._W == np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])).all())
        self.assertTrue((mlp._model[1]._W == np.array([[1, 1, 1, 1], [1, 1, 1, 1]])).all())
        self.assertTrue((mlp.predict([2, 3]) == np.array([[19], [19]])).all())
        mlp.fit([[2, 3]], [[1, 2]])
        self.assertTrue((mlp._model[0]._der_W == np.array([[70, 140, 210], [70, 140, 210], [70, 140, 210]])).all())
        self.assertTrue((mlp._model[1]._der_W == np.array([[36, 216, 216, 216], [34, 204, 204, 204]])).all())
    def test_Layer2(self):
        mlp_arch = [2, Layer(3, activation="linear"), Layer(2, activation="linear")]
        mlp = Model(mlp_arch, 42)
        mlp.compile(loss="mse")
        self.assertTrue(np.allclose(mlp._model[0]._W, np.array([[ 0., 0.49671415, -0.1382643], [0., 0.64768854, 1.52302986], [0., -0.23415337, -0.23413696]])))
        self.assertTrue(np.allclose(mlp._model[1]._W, np.array([[ 0., 1.57921282, 0.76743473, -0.46947439], [0., 0.54256004, -0.46341769, -0.46572975]])))
        self.assertTrue(np.allclose(mlp.predict([2, 3]), np.array([ 5.96400575, -1.85851512])))
        # mlp.fit([[2, 3]], [[1, 2]])
        # self.assertTrue((mlp._model[0]._der_W == np.array([[70, 140, 210], [70, 140, 210], [70, 140, 210]])).all())
        # self.assertTrue((mlp._model[1]._der_W == np.array([[36, 216, 216, 216], [34, 204, 204, 204]])).all())

if __name__ == "__main__":
    unittest.main()
