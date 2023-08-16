import unittest
from neural import Layer
from activations import ident, sigmoid, ident_der, sigmoid_der

class TestLayer(unittest.TestCase):
    def test_Layer1(self):
        layer = Layer(3)
        self.assertEqual(layer._units, 3)
        self.assertEqual(layer._activation, ident)
        self.assertEqual(layer._activation_der, ident_der)
        self.assertEqual(layer._name, None)
        self.assertEqual(layer.get_activation(), ident)
        self.assertEqual(layer.get_activation_der(), ident_der)
        self.assertEqual(layer.get_name(7), "layer_7")

    def test_Layer2(self):
        layer = Layer(4, activation='sigmoid', name='hidden layer')
        self.assertEqual(layer._units, 4)
        self.assertEqual(layer._activation, sigmoid)
        self.assertEqual(layer._activation_der, sigmoid_der)
        self.assertEqual(layer._name, "hidden layer")
        self.assertEqual(layer.get_activation(), sigmoid)
        self.assertEqual(layer.get_activation_der(), sigmoid_der)
        self.assertEqual(layer.get_name(7), "hidden layer")

if __name__ == '__main__':
    unittest.main()