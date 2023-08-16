"""
Train VQ-VAE on MNIST dataset.
"""

import tensorflow as tf

from trainer import Trainer


def main():
    # Load MNIST dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    x_train_scaled = (x_train / 255.0) - 0.5
    x_test_scaled = (x_test / 255.0) - 0.5

    # Create trainer
    trainer = Trainer(num_emb=128, emb_dim=16)

    trainer.compile(optimizer=tf.keras.optimizers.Adam())
    trainer.fit(x_train_scaled, epochs=10, batch_size=128, shuffle=False)
    
    # Save models
    trainer.vqvae.save_weights("vqvae_weights.h5")

if __name__ == "__main__":
    main()
