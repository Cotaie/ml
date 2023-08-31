import numpy as np
import sys


EPSILON = sys.float_info.epsilon
ACTIVATION_POS = LOSS_POS = 0
BIAS_INPUT = ACTIVATION_DER_POS = LOSS_DER_POS = 1
WEIGHTS_INIT_POS = 2
SIGMOID_CLIPPING = 709
SIGMOID_MIDPOINT = 0.5
BIAS_INPUT_NDARRAY = np.array([BIAS_INPUT])
ACTIVATION_NOT_FOUND = "activation function not found, linear is used"
LOSS_NOT_FOUND = "loss function not found, mse is used"

MEAN = 0
VARIANCE = 0.01
LEAKINESS_FACTOR = 0.01