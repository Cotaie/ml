import unittest
from neural.layers import Layer


class TestLayer(unittest.TestCase):
    def test_Layer_only_units(self):
        layer = Layer(3)
        self.assertEqual(layer.units, 3)
        self.assertEqual(layer.activation, None)
        self.assertEqual(layer.name, None)
        self.assertEqual(layer.get_name_or_default(7), "layer_7")
    def test_Layer_all_parameters(self):
        layer = Layer(4, activation='sigmoid', name='hidden layer')
        self.assertEqual(layer.units, 4)
        self.assertEqual(layer.activation, 'sigmoid')
        self.assertEqual(layer.name, "hidden layer")
        self.assertEqual(layer.get_name_or_default(7), "hidden layer")

if __name__ == '__main__':
    unittest.main()