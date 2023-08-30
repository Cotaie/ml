import numpy as np
import sys

EPS = sys.float_info.epsilon
BIAS_INPUT = 1
VARIANCE = 0.01
MEAN = 0
BIAS_INPUT_NDARRAY = np.array([BIAS_INPUT])
ACTIVATION_NOT_FOUND = "activation function not found, linear is used"
LOSS_NOT_FOUND = "loss function not found, mse is used"