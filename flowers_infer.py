import tensorflow as tf

from vqvae import VQVAE
from res_decoder import ResidualDecoder
from res_encoder import Encoder

import matplotlib.pyplot as plt

if __name__ == "__main__":
    latent_dim = 128
    num_emb = 512

    decoder = ResidualDecoder(latent_dim, num_layers=3)
    encoder = Encoder(latent_dim, num_layers=3)

    model = VQVAE(encoder, decoder, num_emb, latent_dim)
    model.build(input_shape=(None, 128, 128, 3))

    model.load_weights("checkpoints_v2/flowers_vqvae_weights_10.h5")
    model.summary()
    
    # Load dataset
    dataset = tf.data.experimental.load("train_dataset")
    dataset = dataset.batch(1)

    batch = next(iter(dataset))
    print(batch.shape)

    batch = tf.reshape(batch, (32, 128, 128, 3))[0:10, ...] / 255.0
    x_hat = model(batch)

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))

    for i in range(10):
        axes[0, i].imshow(batch[i])
        axes[1, i].imshow(x_hat[i])

    # save the figure
    plt.savefig('flowers_infer.png', dpi=300, bbox_inches='tight')
    
