import argparse
import torch

def get_args():
    
    parser = argparse.ArgumentParser(description='ModularizedVAE Training')
    parser.add_argument('--data', required=True, nargs="*", 
                        help='training data path')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='parameters of RMSprop (default: 0.9)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size (default: 128)')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='hidden size of VAE (default: 256)')
    parser.add_argument('--save-intervals', type=int, default=10,
                        help='epoch intervals of saving model (default: 10)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of processes of data loader (default: 4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA')
    parser.add_argument('--prefix', default='ModularizedVAE',
                        help='log file\'s name(i.e. logs/[prefix].log)')
    parser.add_argument('--encoder', default='NoteEventEncoder',
                        help='The encoder to use: NoteEventEncoder, AutoEncoder \
                        see model_architectures/encoder.py to know more. \
                        (default: NoteEventEncoder')
    parser.add_argument('--decoder', default='NoteUnrollingDecoder',
                        help='The decoder to use: NoteEventDecoder, NoteUnrollingDecoder \
                        see model_architectures/decoder.py to know more. \
                        (default: NoteUnrollingDecoder')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
