import numpy as np
from constants import SIGMOID_MIDPOINT

class Evaluate:
    @staticmethod
    def binary_classification(input, output, prediction_fct, loss_fct):
        nr_fails = 0
        sum_loss = 0
        nr_examples = len(output)
        for x, y in zip(input, output):
            pred_y = prediction_fct(x)
            sum_loss += loss_fct(np.array(pred_y), np.array(y))
            if ([0] if pred_y < SIGMOID_MIDPOINT else [1]) != y:
                nr_fails = nr_fails+1
        accuracy = (nr_examples - nr_fails) / nr_examples
        sum_loss /= nr_examples
        return {
            "loss": sum_loss,
            "accuracy": accuracy
        }
    @staticmethod
    def linear(input, output, prediction_fct, loss_fct):
        nr_fails = 0
        sum_loss = 0
        nr_examples = len(output)
        for x, y in zip(input, output):
            pred_y = prediction_fct(x)
            sum_loss += loss_fct(np.array(pred_y), np.array(y))
            if pred_y != y:
                nr_fails = nr_fails+1
        accuracy = (nr_examples - nr_fails) / nr_examples
        sum_loss /= nr_examples
        return {
            "loss": sum_loss,
            "accuracy": accuracy
        }