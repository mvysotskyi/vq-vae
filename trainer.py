"""
Trainer class for VQ-VAE.
"""

import tensorflow as tf
from tensorflow.keras.models import Model

from vqvae import VQVAE


class Trainer(Model):
    def __init__(self, vqvae_model: VQVAE, train_variance: float = 1.0, **kwargs):
        super(Trainer, self).__init__(**kwargs)

        self.train_variance = train_variance

        self.vqvae = vqvae_model
        
        # Metrics
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, x):
        with tf.GradientTape() as tape:
            x_hat = self.vqvae(x)

            recon_loss = tf.reduce_mean(
                tf.math.squared_difference(x, x_hat) / self.train_variance
            )
            total_loss = recon_loss + sum(self.vqvae.losses)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Update metrics
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))
        self.loss_tracker.update_state(total_loss)

        return {
            "loss": self.loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "vq_loss": self.vq_loss_tracker.result()
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_loss_tracker, self.vq_loss_tracker]


if __name__ == "__main__":
    ...
