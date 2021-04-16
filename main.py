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
from utils import get_logger, get_callbacks


# Dataloaders
def get_dataloader(args):
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), lambda x: x.repeat(3, 1, 1)])
        ds_train = MNIST('./data', train=True, download=True, transform=transform)
        ds_validation = MNIST('./data', train=False, download=True, transform=transform)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        dl_validation = DataLoader(ds_validation, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        n_cl, c, h, w = 10, 3, 32, 32

    elif args.dataset == 'cifar10':
        transform = transforms.ToTensor()
        ds_train = CIFAR10('./data', train=True, download=True, transform=transform)
        ds_validation = CIFAR10('./data', train=False, download=True, transform=transform)
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
        'num_embeddings': args.num_embeddings, 
        'beta': args.beta,
        'lr': args.lr,
    }
    model = ImageClassifier(**model_args)
    return model

def get_trainer(args):
    trainer_args = {
        'gpus': 1,
        'max_epochs': args.epochs,
        'callbacks': get_callbacks(args),
        'logger': get_logger(args),
    }
    trainer = pl.Trainer(**trainer_args)
    return trainer

# Main
def main(args):

    # Dataloader
    dls, data_info = get_dataloader(args)

    # Model
    model = get_model(args, data_info)

    # Trainer
    trainer = get_trainer(args)

    # Train
    trainer.fit(model, dls['train'], dls['validation'])


##################################################
# Main
##################################################

if __name__ == '__main__':
    args = get_args(sys.stdin)
    main(args)
