"""
CNN Decoder for the VQ-VAE with Residual Connections.
"""

import tensorflow as tf
from tensorflow.keras import layers


class ResidualDecoder(tf.keras.Model):
    def __init__(self, embedding_dim: int, num_layers: int = 5):
        super().__init__()

        self.embedding_dim = embedding_dim
        
        # Residual block
        self.residual_block = layers.Conv2DTranspose(
            filters=self.embedding_dim,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            name="residual_block"
        )

        self.conv_layers = [
            layers.Conv2DTranspose(
                filters=self.embedding_dim * 2 ** i,
                kernel_size=4,
                strides=2,                
                padding="same",
                activation="relu",
                name=f"conv_{i}"
            )
            for i in range(num_layers - 1)
        ]

        self.conv_out = layers.Conv2DTranspose(
            filters=3,
            kernel_size=3,            
            strides=2,
            padding="same",
            activation="sigmoid",                        
            name="conv_out"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        x = self.residual_block(x) + x

        for conv in self.conv_layers:
            x = conv(x)

        return self.conv_out(x)


if __name__ == '__main__':
    decoder = ResidualDecoder(128, num_layers=3)
    decoder.build(input_shape=(None, 16, 16, 128))
    a = decoder(tf.random.normal((1, 16, 16, 128)))

    decoder.summary()
    print(a.shape)
