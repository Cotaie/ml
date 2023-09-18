import sys


EPSILON = sys.float_info.epsilon
BIAS_INPUT = 1
SIGMOID_CLIPPING = 709
SIGMOID_MIDPOINT = 0.5
ACTIVATION_NOT_FOUND = "activation function not found, linear is used"
LOSS_NOT_FOUND = "loss function not found, mse is used"

MEAN = 0
VARIANCE = 0.01
LEAKINESS_FACTOR = 0.01