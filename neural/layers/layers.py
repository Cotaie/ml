from typing import Callable
from neural._basic.basic import BasicLayer


class Layer(BasicLayer):
    def __init__(self, units: int, activation: str | None = None, kernel_initializer: Callable | None = None, name: str | None = None):
        super().__init__(units)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.name = name

    def get_name_or_default(self, layer_no: int) -> str:
        """
        Retrieves the name of the layer if it's set, otherwise provides a default name based on the layer number.
        Parameters:
        - layer_no (int): The sequence number of the layer, used to generate a default name if the layer's name is not set.
        Returns:
        - str: The name of the layer or a default name in the format "layer_{layer_no}".
        """
        return self.name if self.name is not None else f"layer_{layer_no}"
