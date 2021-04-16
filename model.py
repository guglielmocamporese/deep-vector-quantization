##################################################
# Imports
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, lr_scheduler
from torchvision.models import resnet18

from quantization import VectorQuantized


##################################################
# Classifier Model
##################################################

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes, quantized=False, num_embeddings=512, beta=0.25, lr=3e-4):
        super(ImageClassifier, self).__init__()
        self.quantized = quantized
        self.resnet = nn.Sequential(*list(resnet18().children()))[:-1]
        if self.quantized:
            self.quantize = VectorQuantized(num_embeddings, 512, beta=beta)
        self.clf = nn.Linear(512, num_classes)

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
