"""
Vector Quantizer and VQ-VAE.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class VectorQuantizer(Layer):
    def __init__(self, num_emb: int, emb_dim: int, beta: float = 0.25, **kwargs):
        super().__init__(**kwargs)

        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.beta = beta

        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.emb_dim, self.num_emb),
            initializer="uniform",
            dtype=tf.float32,
            trainable=True
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        original_shape = tf.shape(inputs)
        flatten_inputs = tf.reshape(inputs, (-1, self.emb_dim))

        # Find the nearest embedding for each input vector.
        embedding_indices = self.__nearest_embedding(flatten_inputs)
        encodings = tf.one_hot(embedding_indices, self.num_emb)
        quantized = tf.matmul(encodings, tf.transpose(self.embeddings))

        # Reshape the quantized vector to match the original input shape.
        quantized = tf.reshape(quantized, original_shape)

        # Compute the commitment loss.
        commitment_loss = tf.reduce_mean(
            tf.square(inputs - tf.stop_gradient(quantized))
        )

        # Compute the embedding loss.
        emb_loss = tf.reduce_mean(
            tf.square(tf.stop_gradient(quantized) - inputs)
        )

        # Compute the loss.
        self.add_loss(emb_loss + self.beta * commitment_loss)
    
        # Straight-through estimator.
        return inputs + tf.stop_gradient(quantized - inputs)


    def __nearest_embedding(self, flatten_inputs: tf.Tensor) -> tf.Tensor:
        """
        Find the nearest embedding for each input vector.

        flatten_inputs: (batch_size * height * width, emb_dim)
        """
        # Compute the L2 distance between each input vector and each embedding.
        distances = (
            tf.reduce_sum(tf.square(flatten_inputs), axis=1, keepdims=True) + 
            tf.reduce_sum(tf.square(self.embeddings), axis=0) -
            2 * tf.matmul(flatten_inputs, self.embeddings)
        )

        return tf.argmin(distances, axis=-1)

    @staticmethod
    @tf.custom_gradient
    def __straight_through_estimator(inputs: tf.Tensor):
        return inputs, lambda dy: dy

if __name__ == "__main__":
    vq = VectorQuantizer(10, 4)
    vq(tf.random.uniform((5, 12, 12, 4)))
