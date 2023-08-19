"""
CNN Encoder for the VQ-VAE.
"""

import tensorflow as tf
from tensorflow.keras import layers


class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim: int, num_layers: int = 5):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.conv_layers = [
            layers.Conv2D(
                filters=64 * 2 ** i,
                kernel_size=4,
                strides=2,                
                padding="same",
                activation="relu",
                name=f"conv_{i}"
            )
            for i in range(num_layers - 1)
        ]

        self.conv_out = layers.Conv2D(
            filters=self.embedding_dim,
            kernel_size=4,            
            strides=2,
            padding="same",
            activation="relu",                        
            name="conv_out"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs

        for conv in self.conv_layers:
            x = conv(x)

        return self.conv_out(x)

if __name__ == '__main__':
    encoder = Encoder(128, num_layers=3)
    encoder.build(input_shape=(None, 128, 128, 3))
    a = encoder(tf.random.normal((1, 128, 128, 3)))

    encoder.summary()
    print(a.shape)
