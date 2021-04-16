##################################################
# Imports
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################
# Vector Quantized
##################################################

class VectorQuantized(nn.Module):
    """
    Inspired by VQVAE: https://arxiv.org/abs/1711.00937
    """
    def __init__(self, num_emb, in_dim, ema=False, decay=0.99, beta=0.25):
        super(VectorQuantized, self).__init__()
        self.num_emb = num_emb
        self.in_dim = in_dim
        self.ema = ema
        self.decay = decay
        self.beta = beta
        self.emb = nn.Embedding(num_emb, in_dim)

    def forward(self, x_in):
        """
        Inputs:
            x: tensor of shape [B, C, ...]

        Outputs:
            x_q: tensor of shape [B, C, ...] (same as x)
            idxs: tensor of shape [B, ...]
            vq_loss: scalar
        """
        x = x_in.unsqueeze(1).transpose(1, -1).squeeze(-1) # [B, ..., C]
        x_shape = x.shape
        x_flat = x.reshape(-1, self.in_dim) # [B * ..., C]

        # Fla
        dist = torch.cdist(x_flat.unsqueeze(0), self.emb.weight.unsqueeze(0))[0] # [B  * ..., num_emb]
        idxs = dist.argmin(-1).view(x_shape[:-1]) # [B, ...]
        x_q = self.emb(idxs) # [B, ..., C]

        q_loss = F.mse_loss(x_q, x.detach())
        e_loss = F.mse_loss(x, x_q.detach())
        vq_loss = q_loss + self.beta * e_loss

        # TODO: implement ema update for the embeddings.

        x_q = x_q.unsqueeze(1).transpose(1, -1).squeeze(-1) # [B, C, ...]
        x_q = x_in + (x_q - x_in).detach() # Copy gradient
        return x_q, idxs, vq_loss


if __name__ == '__main__':
    x = torch.randn(10, 256, 50)

    tq = VectorQuantized(512, 256)

    x_q, idxs, vq_loss = tq(x)
    print(x_q.shape, idxs.shape, vq_loss)
