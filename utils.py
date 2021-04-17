##################################################
# Imports
##################################################

import pytorch_lightning as pl
import os
import torch
import random
import numpy as np

def get_callbacks(args):

    # Callbacks
    model_ckpt = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=None,
        filename='best',
        monitor='valid_acc' if args.task == 'classification' else 'valid_rec',
        verbose=True,
        save_last=True,
        mode='max' if args.task == 'classification' else 'min',    
    )
    model_ckpt.CHECKPOINT_NAME_LAST = '{epoch}-{step}'
    callbacks = [model_ckpt]
    return callbacks

def get_logger(args):

    # Logger
    logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir='./tmp',
        name=f'{args.dataset}_{args.task}',
    )
    return logger
