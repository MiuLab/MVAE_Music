import argparse
import random

from model import ModularizedVAE
from midi_utils import dump_midi, read_midi_files
from loader import MusicDataset

import numpy as np
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='ModularizedVAE generating')
parser.add_argument('-l', '--length', type=int, default=100,
                    help='number of note to generate (default: 100)')
parser.add_argument('-o', '--output', default='output.mid',
                    help='output path (default: output.mid)')
parser.add_argument('-m', '--model', required=True,
                    help='model parameter\'s path')
parser.add_argument('--seed', type=int, default=9487,
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if __name__ == '__main__':
    
    model = torch.load(args.model)
    vae = ModularizedVAE(**model['model_args'])
    vae.load_state_dict(model['state_dict'])
    vae.eval()

    #torch.manual_seed(args.seed)

    if args.cuda:
        vae = vae.cuda()
    output = vae.sample(length=args.length)
    dump_midi(output, model['note_sets'], args.output)
