import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ, ResidualSimVQ


FIRST_PROJECT_DIM = 2048
SECOND_PROJECT_DIM = 1024


class TruthXVAE(nn.Module):
    def get_basic_block(self, d1, d2):
        return nn.Sequential(
            nn.Linear(d1, d2),
        )

    def __init__(self, vae_config):
        super().__init__()
        embedding_dim = vae_config["embedding_dim"]
        model_type = vae_config['vq_type']
        self.encoder = nn.Sequential(
            self.get_basic_block(embedding_dim, FIRST_PROJECT_DIM),
            self.get_basic_block(FIRST_PROJECT_DIM, SECOND_PROJECT_DIM),
        )
        if "ResidualVQ" in model_type:
            self.vqvae = ResidualVQ(
                dim = SECOND_PROJECT_DIM,
                num_quantizers = vae_config['num_quantizers'],      # specify number of quantizers
                codebook_size = vae_config['codebook_size'],    # codebook size
            )
        elif "SimVQ" in model_type:
            self.vqvae = ResidualSimVQ(
                dim = SECOND_PROJECT_DIM,
                num_quantizers = vae_config['num_quantizers'],
                codebook_size = vae_config['codebook_size'],
                rotation_trick = True,  # use rotation trick from Fifty et al.
            )
        self.decoder = nn.Sequential(
            self.get_basic_block(SECOND_PROJECT_DIM, FIRST_PROJECT_DIM),
            self.get_basic_block(FIRST_PROJECT_DIM, embedding_dim),
        )
    
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices, cmt_loss = self.vqvae(z_e)
        out = self.decoder(z_q)
        return out, indices, cmt_loss
        