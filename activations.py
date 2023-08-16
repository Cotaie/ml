import numpy as np

def ident(z):
    return z

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return max(0, z)

map_activations = {
    'linear': ident,
    'sigmoid': sigmoid,
    'relu': relu
}

def ident_der(z):
    return 1

def sigmoid_der(z):
    sigmoid_value = sigmoid(z)
    return sigmoid_value * (1.0 - sigmoid_value)

def relu_der(z):
    return 1.0 if z > 0 else 0.0

map_activations_der = {
    'linear': ident_der,
    'sigmoid': sigmoid_der,
    'relu': relu_der
}
