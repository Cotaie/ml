import numpy as np

def ident(z):
    return z

def sigmoid(z):
    return 1 / (1 + np.exp(-z))