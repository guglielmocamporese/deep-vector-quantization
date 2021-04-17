##################################################
# Imports
##################################################

import pytorch_lightning as pl
import os
import torch
import random
import numpy as np
import math


def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

"""
These ramps/decays follow DALL-E Appendix A.2 Training https://arxiv.org/abs/2102.12092
"""
class DecayTemperature(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The relaxation temperature τ is annealed from 1 to 1/16 over the first 150,000 updates.
        t = cos_anneal(0, 150000, 1.0, 1.0/16, trainer.global_step)
        pl_module.quantize.temp_init = t

class RampBeta(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The KL weight β is increased from 0 to 6.6 over the first 5000 updates
        # "We divide the overall loss by 256 × 256 × 3, so that the weight of the KL term
        # becomes β/192, where β is the KL weight."
        # TODO: OpenAI uses 6.6/192 but kinda tricky to do the conversion here... about 5e-4 works for this repo so far... :\
        t = cos_anneal(0, 5000, 0.0, 5e-4, trainer.global_step)
        pl_module.quantize.kld_scale = t

class DecayLR(pl.Callback):
    def __init__(self, lr_init=3e-4, lr_end=1.25e-6):
        super(DecayLR, self).__init__()
        self.lr_init = lr_init
        self.lr_end = lr_end

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        t = cos_anneal(0, 120000, self.lr_init, self.lr_end, trainer.global_step)
        for g in pl_module.optimizer.param_groups:
            g['lr'] = t

def get_callbacks(args):

    # Model checkpoint
    model_ckpt = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=None,
        filename='best',
        monitor='valid_acc' if args.task == 'classification' else 'valid_rec',
        verbose=True,
        save_last=True,
        mode='max' if args.task == 'classification' else 'min',    
    )
    model_ckpt.CHECKPOINT_NAME_LAST = '{epoch}-{step}'
    callbacks = [
        model_ckpt,
        DecayLR(lr_init=args.lr),
    ]
    if args.vq_mode == 'gumbel':
        callbacks += [DecayTemperature()]#, RampBeta()]
    return callbacks

def get_logger(args):

    # Logger
    logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir='./tmp',
        name=f'{args.dataset}_{args.task}',
    )
    return logger
