##################################################
# Imports
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, lr_scheduler

from models import resnet, quantization


##################################################
# Classifier Model
##################################################

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes, quantized=False, lr=3e-4, dropout=0.2, *args, **kwargs):
        super(ImageClassifier, self).__init__()
        self.quantized = quantized
        self.resnet, feat_dim = self._make_backbone()
        if self.quantized:
            self.quantize = quantization.VectorQuantized(in_dim=feat_dim, *args, **kwargs)
        self.clf = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def _make_backbone(self):
        _resnet = resnet.ResNet18(use_as_backone=True)
        backbone = nn.Sequential(
            _resnet, 
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        feat_dim = _resnet.feat_dim
        return backbone, feat_dim

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        if self.quantized:
            x, idxs, vq_loss = self.quantize(x)
        logits = self.clf(x)
        out = {
            'logits': logits,
            'probs': F.softmax(logits, -1),
            'x': x,
            'idxs': idxs if self.quantized else None,
            'vq_loss': vq_loss if self.quantized else None,
        }
        return out

    def training_step(self, batch, idx_batch, part='train'):
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds['logits'], y)
        acc = (1.0 * (preds['probs'].argmax(-1) == y)).mean()

        self.log(f'{part}_loss', loss, prog_bar=True)
        self.log(f'{part}_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, idx_batch):
        return self.training_step(batch, idx_batch, part='valid')

    def test_step(self, batch, idx_batch):
        return self.training_step(batch, idx_batch, part='test')

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), 3e-4)
        scheduler = {
            'scheduler': lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True),
            'monitor': 'valid_acc',
        }
        return [optimizer], [scheduler]

def get_model(args, data_info):
    model_args = {
        'num_classes': data_info['num_classes'], 
        'quantized': args.vq_mode in ['vq', 'vq_ema', 'gumbel'], 
        'num_emb': args.num_embeddings, 
        'beta': args.beta,
        'lr': args.lr,
        'dropout': args.dropout,
        'vq_mode': args.vq_mode,
        'decay': args.decay,
        'beta': args.beta,
        'temp_init': args.temp_init,
        'straight_through': args.straight_through,
    }
    model = ImageClassifier(**model_args)
    if len(args.model_checkpoint) > 0:
        model = model.load_from_checkpoint(args.model_checkpoint, **model_args)
        print('Loaded checkpoints at "{args.model_checkpoint}"')
    return model

# Debug...
if __name__ == '__main__':
    x = torch.randn(10, 3, 512, 512)
    model = ImageClassifier(10, quantized=False)

    out = model(x)
    for k, v in out.items():
        if torch.is_tensor(v):
            print(k, v.shape)
        else:
            print(k, v)