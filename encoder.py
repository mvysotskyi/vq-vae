"""
Encoder model for VQ-VAE.
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential


class Encoder(Model):
    def __init__(self, embedding_dim: int):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim

        self.conv_1 = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")
        self.conv_3 = tf.keras.layers.Conv2D(embedding_dim, 1, padding="same")

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)

        return x

if __name__ == "__main__":
    encoder = Encoder(64)
    encoder.build(input_shape=(128, 28, 28, 1))
    encoder.summary()
    
    a = encoder(tf.random.normal((1, 28, 28, 1)))
    print(a.shape)
