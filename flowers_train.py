"""
Training VQ-VAE on the Flowers dataset.
"""

import tensorflow as tf

from res_encoder import Encoder
from res_decoder import ResidualDecoder
from vqvae import VQVAE

from trainer import Trainer


def main():
    # Load MNIST dataset
    train_dataset = tf.data.experimental.load("train_dataset")
    train_dataset = train_dataset.map(lambda x: tf.cast(x, tf.float32) / 255.0)
    
    #train_variance = tf.math.reduce_variance(train_dataset, axis=0)
    train_variance = 1.0

    # Create trainer
    model = VQVAE(
        Encoder(128, num_layers=3),
        ResidualDecoder(128, num_layers=3),
        512,
        128
    )

    trainer = Trainer(model, train_variance)
    
    trainer.build(input_shape=(None, 128, 128, 3))
    trainer.summary()
    trainer.vqvae.load_weights("flowers_vqvae_weights_8.h5")

    trainer.compile(optimizer=tf.keras.optimizers.Adam())
    trainer.fit(train_dataset, epochs=2, batch_size=32, shuffle=False)
    
    # Save models
    trainer.vqvae.save_weights("flowers_vqvae_weights_10.h5")

if __name__ == "__main__":
    main()
