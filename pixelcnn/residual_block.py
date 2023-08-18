"""
ResidualBlock layer implementation.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, add as layers_add

from pixel_convolution import PixelConv


class ResidualBlock(Layer):
    def __init__(self, num_filters: int, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.in_conv = Conv2D(filters=num_filters, kernel_size=1, activation="relu")
        self.out_conv = Conv2D(filters=num_filters, kernel_size=1, activation="relu")

        self.pixel_conv = PixelConv(
            mask_type="B",
            filters=num_filters // 2,
            kernel_size=3,
            padding="same",
            activation="relu",
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Call layer.
        """
        x = self.in_conv(inputs)
        x = self.pixel_conv(x)
        x = self.out_conv(x)

        return layers_add([inputs, x])


if __name__ == "__main__":
    ...
