import unittest
from neural import Layer

class TestLayer(unittest.TestCase):
    def test_Layer_only_units(self):
        layer = Layer(3)
        self.assertEqual(layer._units, 3)
        self.assertEqual(layer._activation, None)
        self.assertEqual(layer._name, None)
        self.assertEqual(layer.get_units(), 3)
        self.assertEqual(layer.get_activation(), None)
        self.assertEqual(layer.get_name(7), "layer_7")
    def test_Layer_all_parameters(self):
        layer = Layer(4, activation='sigmoid', name='hidden layer')
        self.assertEqual(layer._units, 4)
        self.assertEqual(layer._activation, 'sigmoid')
        self.assertEqual(layer._name, "hidden layer")
        self.assertEqual(layer.get_units(), 4)
        self.assertEqual(layer.get_activation(), 'sigmoid')
        self.assertEqual(layer.get_name(7), "hidden layer")

if __name__ == '__main__':
    unittest.main()