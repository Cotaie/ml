import unittest
import numpy as np
from neural import Layer, Model

class TestModel(unittest.TestCase):
    def test_fit(self):
        mlp = Model([2, Layer(3, activation="linear"), Layer(2, activation="linear")], 42)
        mlp.compile(loss="mse")
        if not np.allclose(mlp._model[0]._W, np.array([[ 0., 0.49671415, -0.1382643], [0., 0.64768854, 1.52302986], [0., -0.23415337, -0.23413696]])):
            self.fail("Layer 0 weights do not match expectations")
        if not np.allclose(mlp._model[1]._W, np.array([[ 0., 1.57921282, 0.76743473, -0.46947439], [0., 0.54256004, -0.46341769, -0.46572975]])):
            self.fail("Layer 1 weights do not match expectations")
        if not np.allclose(mlp.predict([2, 3]), np.array([ 5.96400575, -1.85851512])):
            self.fail("Prediction fail")
        mlp.fit([[2, 3]], [[1, 2]])
        if not np.allclose(mlp._model[1]._der_W, np.array([[9.92801154021736, 5.74469892877829, 58.2224926777, -11.6228980416958],[-7.71703024481321, -4.46534688251959, -45.2562665849187, 9.03446328167574]])):
            self.fail("Layer 1 der W fail")
        if not np.allclose(mlp._model[0]._der_W, np.array([[11.4914908631121, 22.9829817262243, 34.4744725893364], [11.1953091855151, 22.3906183710301, 33.5859275565452], [-1.06689659509721, -2.13379319019443, -3.20068978529164]])):
            self.fail("Layer 0 der W fail")
if __name__ == "__main__":
    unittest.main()
