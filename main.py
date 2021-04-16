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


# Dataloaders
def get_dataloaders(args):
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), lambda x: x.repeat(3, 1, 1)])
        ds_train = MNIST('./data', train=True, download=True, transform=transform)
        ds_validation = MNIST('./data', train=False, download=True, transform=transform)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        dl_validation = DataLoader(ds_validation, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        n_cl, c, h, w = 10, 3, 32, 32

    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_validation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        ds_train = CIFAR10('./data', train=True, download=True, transform=transform_train)
        ds_validation = CIFAR10('./data', train=False, download=True, transform=transform_validation)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        dl_validation = DataLoader(ds_validation, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        n_cl, c, h, w = 10, 3, 32, 32

    else:
        raise Exception(f'Error. Dataset "{args.dataset}" is not supported.')

    data_info = {
        'num_classes': n_cl,
        'height': h,
        'width': w,
        'channels': c,
    }
    dls_dict = {
        'train': dl_train,
        'validation': dl_validation,
    }
    return dls_dict, data_info


# Model
def get_model(args, data_info):
    model_args = {
        'num_classes': data_info['num_classes'], 
        'quantized': args.quantized, 
        'num_emb': args.num_embeddings, 
        'in_dim': 512,
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
    dls, data_info = get_dataloaders(args)

    # Model
    model = get_model(args, data_info)

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
