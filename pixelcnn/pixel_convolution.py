"""
PixelConv layer implementation.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D


class PixelConv(Layer):
    def __init__(self, mask_type: str = "A", **kwargs):
        assert mask_type in ["A", "B"], "mask_type must be either 'A' or 'B'"
        super(PixelConv, self).__init__()

        self.mask_type = mask_type
        self.conv_layer = Conv2D(**kwargs)
        self.mask = None

    def _generate_mask(self) -> tf.Tensor:
        """
        Generate mask for convolution.
        """
        kernel_shape = self.conv_layer.kernel.shape
        kernel_len = kernel_shape[0] * kernel_shape[1]
        center = kernel_len // 2 + (1 if self.mask_type == "B" else 0) 
        
        mask = tf.constant([1.0] * center + [0.0] * (kernel_len - center), dtype=tf.float32)
        mask = tf.reshape(mask, [kernel_shape[0], kernel_shape[1], 1, 1])
        mask = tf.tile(mask, [1, 1] + kernel_shape[2:])
                
        return mask

    def build(self, input_shape: tuple) -> None:
        """
        Build layer.
        """
        self.conv_layer.build(input_shape)
        self.mask = self._generate_mask()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Call layer.
        """
        self.conv_layer.kernel.assign(self.conv_layer.kernel * self.mask)
        return self.conv_layer(inputs)

if __name__ == "__main__":
    pixel_conv = PixelConv(
        filters=32,
        kernel_size=5,
        strides=1,
        padding="same",
        mask_type="A",
        activation="relu",
    )

    pixel_conv.build(input_shape=(None, 28, 28, 1))
    print(pixel_conv._generate_mask().shape)
