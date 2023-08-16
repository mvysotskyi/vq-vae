"""
Decoder for VQ-VAE.
"""

import tensorflow as tf
from tensorflow.keras.models import Model


class Decoder(Model):
    def __init__(self, embedding_dim: int):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim

        self.deconv_1 = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")
        self.deconv_2 = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")
        self.deconv_3 = tf.keras.layers.Conv2DTranspose(1, 3, padding="same")

    def call(self, inputs):
        x = self.deconv_1(inputs)
        x = self.deconv_2(x)
        x = self.deconv_3(x)

        return x

if __name__ == "__main__":
    decoder = Decoder(64)
    decoder.build(input_shape=(128, 7, 7, 64))
    decoder.summary()
    
    a = decoder(tf.random.normal((5, 7, 7, 64)))
    print(a.shape)
