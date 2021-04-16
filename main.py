##################################################
# Imports
##################################################

import pytorch_lightning as pl
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import sys

from model import ImageClassifier
from config import get_args
import utils
import dataloaders
import model


def get_trainer(args):
    trainer_args = {
        'gpus': 1,
        'max_epochs': args.epochs,
        'callbacks': utils.get_callbacks(args),
        'logger': utils.get_logger(args),
        'deterministic': True,
    }
    trainer = pl.Trainer(**trainer_args)
    return trainer

# Main
def main(args):

    # Dataloader
    dls, data_info = dataloaders.get_dataloaders(args)

    # Model
    model = model.get_model(args, data_info)

    # Trainer
    trainer = get_trainer(args)

    if args.mode in ['train', 'training']:
    
        # Train
        trainer.fit(model, dls['train'], dls['validation'])

        # Validate
        trainer.test(model=None, test_loaders=dls['validate'])

    elif args.mode == ['validate', 'validation']:

        trainer.test(model, dls['validation'])

    else:
        raise Exception(f'Error. Mode "{args.mode}" is not supported.')


##################################################
# Main
##################################################

if __name__ == '__main__':
    
    # Imports
    import json
    
    # Args
    args = get_args(sys.stdin)
    print(json.dumps(vars(args), indent=4))

    # Repreoducibility
    pl.seed_everything(args.seed)

    # Run main
    main(args)
