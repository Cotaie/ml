from .constants import SIGMOID_MIDPOINT

class Evaluate:
    @staticmethod
    def binary_classification(pred_y, output):
        return ([0] if pred_y < SIGMOID_MIDPOINT else [1]) != output
    @staticmethod
    def linear(pred_y, output):
        return pred_y != output