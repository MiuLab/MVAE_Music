import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseVAE(nn.Module):
    '''
    Base Variational Autoencoder.

    Arguments:
    encoder -- torch.nn.Module object.
    decoder -- torch.nn.Module object.
    '''
    def __init__(
            self,
            encoder,
            decoder):
        super(BaseVAE, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
    
    def encode(self, x):
        return self._encoder(x)

    def decode(self, z):
        return self._decoder(z)

    def reconstruct_loss(self, predict, target):
        return F.cross_entropy(predict, target)

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))

