##################################################
# Imports
##################################################

import pytorch_lightning as pl
import torch
import sys

from config import get_args
from utils import get_callbacks, get_logger
from dataloaders import get_dataloaders
from models import classifier


def get_trainer(args):
    trainer_args = {
        'gpus': 1,
        'max_epochs': args.epochs,
        'callbacks': get_callbacks(args),
        'logger': get_logger(args),
        'deterministic': True,
    }
    trainer = pl.Trainer(**trainer_args)
    return trainer

# Main
def main(args):

    # Dataloader
    dls, data_info = get_dataloaders(args)

    # Model
    model = classifier.get_model(args, data_info)

    # Trainer
    trainer = get_trainer(args)

    if args.mode in ['train', 'training']:
    
        # Train
        trainer.fit(model, dls['train'], dls['validation'])

        # Validate
        trainer.test(model=None, test_loaders=dls['validate'])

    elif args.mode in ['validate', 'validation']:

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
