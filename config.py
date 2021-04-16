##################################################
# Imports
##################################################

import argparse

def get_args(stdin):
    parser = argparse.ArgumentParser(stdin)
    parser.add_argument('--dataset', type=str, help='The dataset.')
    parser.add_argument('--batch_size', type=int, default=256, help='The batch size.')
    parser.add_argument('--lr', type=float, default=3e-4, help='The learning rate.')
    parser.add_argument('--epochs', type=int, default=30, help='The number of epochs for the learning process.')
    parser.add_argument('--quantized', action='store_true', help='Use quantized network.')
    parser.add_argument('--beta', type=float, default=0.25, help='Loss weight (see VQVAE paper).')
    parser.add_argument('--decay', type=float, default=0.99, help='Decay factor in the EMA embeddings learning (see VQVAE paper).')
    parser.add_argument('--ema', action='store_true', help='Use exponential moving average for learning the embeddings during quantization.')
    parser.add_argument('--num_embeddings', type=int, default=512, help='Number of embeddings for the quantization layers.')
    parser.add_argument('--seed', type=int, default=1234, help='The random seed, for reproduciblity.')

    args = parser.parse_args()
    return args
