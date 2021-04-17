##################################################
# Imports
##################################################

from torchvision import transforms
from torchvision.datasets import MNIST, SVHN, CIFAR10, CIFAR100
from torch.utils.data import DataLoader

def get_datasets(args):

    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), lambda x: x.repeat(3, 1, 1)])
        ds_train = MNIST('./data', train=True, download=True, transform=transform)
        ds_validation = MNIST('./data', train=False, download=True, transform=transform)
        n_cl, c, h, w = 10, 3, 32, 32

    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_validation = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        ds_train = CIFAR10('./data', train=True, download=True, transform=transform_train)
        ds_validation = CIFAR10('./data', train=False, download=True, transform=transform_validation)
        n_cl, c, h, w = 10, 3, 32, 32

    else:
        raise Exception(f'Error. Dataset "{args.dataset}" is not supported.')

    data_info = {
        'num_classes': n_cl,
        'height': h,
        'width': w,
        'channels': c,
    }
    dss_dict = {
        'train': ds_train,
        'validation': ds_validation,
    }
    return dss_dict, data_info

def get_dataloaders(args):

    # Datasets
    dss_dict, data_info = get_datasets(args)

    # Dataloaders
    dls_dict = {
        'train': DataLoader(dss_dict['train'], batch_size=args.batch_size, shuffle=True, pin_memory=True),
        'validation':  DataLoader(dss_dict['validation'], batch_size=args.batch_size, shuffle=False, pin_memory=True),
    }
    return dls_dict, data_info
