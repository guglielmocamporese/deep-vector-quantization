##################################################
# Imports
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid
from torch.optim import Adam, lr_scheduler

from models import resnet, quantization


##################################################
# Residual layers
##################################################

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


##################################################
# Encoder
##################################################

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)


##################################################
# Decoder
##################################################

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, x):
        x = self._conv_1(x)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


##################################################
# Auto Encoder
##################################################

class AutoEncoder(pl.LightningModule):
    def __init__(self, quantized=False, q_dim=64, lr=3e-4, backbone='resnet18', *args, **kwargs):
        super(AutoEncoder, self).__init__()
        self.quantized = quantized
        self.lr = lr
        self.enc, feat_dim = self._make_backbone(backbone)
        self.proj = nn.Conv2d(feat_dim, q_dim, kernel_size=1, stride=1)
        if self.quantized:
            self.quantize = quantization.VectorQuantized(in_dim=q_dim, *args, **kwargs)
        self.dec = Decoder(q_dim, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32)

    def _make_backbone(self, backbone):
        if backbone == 'resnet18':
            _resnet = resnet.ResNet18(use_as_backone=True)
            backbone = nn.Sequential(
                _resnet, 
            )
            feat_dim = _resnet.feat_dim
        elif backbone == 'deepmind':
            backbone = Encoder(3, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32)
            feat_dim = 128
        else:
            raise Exception(f'Error. Bckbone "{backbone}" is not supported.')

        return backbone, feat_dim

    def forward(self, x):
        x = self.enc(x)
        x = self.proj(x)
        if self.quantized:
            x, idxs, vq_loss, perplexity = self.quantize(x)
        logits = self.dec(x)
        x_hat = torch.sigmoid(logits)
        out = {
            'x_hat': x_hat,
            'idxs': idxs if self.quantized else None,
            'vq_loss': vq_loss if self.quantized else None,
            'perplexity': perplexity if self.quantized else None,
        }
        return out

    def training_step(self, batch, idx_batch, part='train'):
        x, _ = batch
        preds = self(x)
        loss_rec = F.binary_cross_entropy(preds['x_hat'], x)
        loss = loss_rec
        if self.quantized:
            loss += preds['vq_loss']

        self.log(f'{part}_rec', loss_rec, prog_bar=True)
        if self.quantized:
            self.log(f'{part}_vq_loss', preds['vq_loss'], prog_bar=True)
            self.log(f'{part}_perplexity', preds['perplexity'], prog_bar=True)
        if (idx_batch == 0) and (part == 'valid'):
            tb_logger = self.logger.experiment
            x_in_grid = make_grid(x[:25], nrow=5, padding=0)
            x_hat_grid = make_grid(preds['x_hat'][:25], nrow=5, padding=0)
            tb_logger.add_image(f'{part}_img_in', x_in_grid, self.current_epoch)
            tb_logger.add_image(f'{part}_img_rec', x_hat_grid, self.current_epoch)

        return loss

    def validation_step(self, batch, idx_batch):
        return self.training_step(batch, idx_batch, part='valid')

    def test_step(self, batch, idx_batch):
        return self.training_step(batch, idx_batch, part='test')

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), self.lr)
        scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True),
            'monitor': 'valid_rec',
        }
        return [optimizer], [scheduler]

def get_model(args, data_info):
    model_args = {
        'quantized': args.vq_mode in ['vq', 'vq_ema', 'gumbel'],
        'q_dim': 64,
        'num_emb': args.num_embeddings, 
        'beta': args.beta,
        'lr': args.lr,
        'vq_mode': args.vq_mode,
        'decay': args.decay,
        'beta': args.beta,
        'temp_init': args.temp_init,
        'straight_through': args.straight_through,
        'backbone': args.backbone,
    }
    model = AutoEncoder(**model_args)
    return model
