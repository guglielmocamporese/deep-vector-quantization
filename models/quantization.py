##################################################
# Imports
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


##################################################
# Vector Quantized
##################################################

class VectorQuantized(nn.Module):
    """
    General class for quantized vectors.
    """
    def __init__(self, vq_mode='vq', *args, **kwargs):
        super(VectorQuantized, self).__init__()
        self.vq = self._get_vq(vq_mode, *args, **kwargs)

    def _get_vq(self, vq_mode, *args, **kwargs):
        if vq_mode == 'vq':
            return VQ(*args, **kwargs)
        elif vq_mode == 'vq_ema':
            return VQEMA(*args, **kwargs)
        elif vq_mode == 'gumbel':
            return VQGumbel(*args, **kwargs)
        else:
            raise Exception(f'Error. Mode "{vq_mode}" is not supported.')

    def forward(self, x):
        return self.vq(x)


class VQ(nn.Module):
    """
    Inspired by VQVAE
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937
    """
    def __init__(self, num_emb, in_dim, beta=0.25, *args, **kwargs):
        super(VQ, self).__init__()
        self.num_emb = num_emb
        self.in_dim = in_dim
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

        dist = torch.cdist(x_flat.unsqueeze(0), self.emb.weight.unsqueeze(0))[0] # [B  * ..., num_emb]
        idxs = dist.argmin(-1).view(x_shape[:-1]) # [B, ...]
        x_q = self.emb(idxs) # [B, ..., C]

        q_loss = F.mse_loss(x_q, x.detach())
        e_loss = F.mse_loss(x, x_q.detach())
        vq_loss = q_loss + self.beta * e_loss

        x_q = x_q.unsqueeze(1).transpose(1, -1).squeeze(-1) # [B, C, ...]
        x_q = x_in + (x_q - x_in).detach() # Copy gradient
        return x_q, idxs, vq_loss


class VQEMA(nn.Module):
    """
    Inspired by VQVAE
    Neural Discrete Representation Learning, van den Oord et al. 2017
    https://arxiv.org/abs/1711.00937
    """
    def __init__(self, num_emb, in_dim, decay=0.99, beta=0.25, *args, **kwargs):
        super(VQEMA, self).__init__()
        self.num_emb = num_emb
        self.in_dim = in_dim
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


class VQGumbel(nn.Module):
    """
    Inspired by the Gumbel-Softmax trick.
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_emb, in_dim, temp_init=1.0, straight_through=False, *args, **kwargs):
        super(VQGumbel, self).__init__()
        self.num_emb = num_emb
        self.in_dim = in_dim
        self.temp_init = temp_init
        self.straight_through = straight_through
        self.emb = nn.Embedding(num_emb, in_dim)
        self.proj = nn.Linear(in_dim, num_emb)

    def forward(self, x_in, temp=None):
        """
        Inputs:
            x: tensor of shape [B, C, ...]
            temp: scalar, temperature for the gumbel-softmax function

        Outputs:
            x_q: tensor of shape [B, C, ...] (same as x)
            idxs: tensor of shape [B, ...]
            vq_loss: scalar
        """
        if temp is None:
            temp = self.temp_init
        x = x_in.unsqueeze(1).transpose(1, -1).squeeze(-1) # [B, ..., C]
        x_shape = x.shape
        x_flat = x.reshape(-1, self.in_dim) # [B * ..., C]
        x_logits = self.proj(x_flat) # [B * ..., num_emb]
        hard = self.straight_through if self.training else True
        x_soft = F.gumbel_softmax(x_logits, tau=temp, dim=1, hard=hard) # [B * ..., num_emb]

        idxs = x_soft.argmax(1) # [B * ...]
        x_q = torch.mm(x_soft, self.emb.weight) # [B * ..., C]
        q_y = F.softmax(x_logits, 1) # [B * ..., num_emb]
        vq_loss = (q_y * (torch.log_softmax(x_logits, 1) + math.log(self.num_emb))).sum(1).mean(0) # scalar
        
        # Reshape to original
        x_q = x_q.reshape(x_shape).unsqueeze(1).transpose(1, -1).squeeze(-1)
        idxs = idxs.reshape(x_shape[:-1])
        return x_q, idxs, vq_loss


# Debug...
if __name__ == '__main__':
    x = torch.randn(10, 256, 50)

    tq = VectorQuantized(512, 256)

    x_q, idxs, vq_loss = tq(x)
    print(x_q.shape, idxs.shape, vq_loss)