"""
VQ-VAE model.
"""

from tensorflow.keras.models import Model

from encoder import Encoder
from decoder import Decoder

from vector_quantizer import VectorQuantizer


class VQVAE(Model):
    def __init__(self, num_emb: int = 128, emb_dim: int = 16, **kwargs):
        super(VQVAE, self).__init__(**kwargs)

        self.num_emb = num_emb
        self.emb_dim = emb_dim

        self.encoder = Encoder(emb_dim)
        self.decoder = Decoder(emb_dim)

        self.vector_quantizer = VectorQuantizer(num_emb, emb_dim)

    def call(self, inputs):
        z_e = self.encoder(inputs)
        z_q = self.vector_quantizer(z_e)
        x_hat = self.decoder(z_q)

        return x_hat

if __name__ == "__main__":
    model = VQVAE(1024, 64)
    model.build(input_shape=(128, 28, 28, 1))

    model.summary()
