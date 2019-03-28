import torch
import torch.nn as nn
import torch.nn.functional as F


class NoteEventEncoder(nn.Module):
    '''
    NoteEventEncoder.
    Encode tuple (timing, duration, pitch) into a latent vector z.

    Arguments:
    input_size -- 3-tuple for (timing, duration, pitch)'s size
    hidden_size -- hidden_size of grus
    z_size -- dimension of latent vector.
    '''
    def __init__(
            self,
            input_size,
            hidden_size):
        super(NoteEventEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.NE_grus = nn.ModuleList([
            nn.GRU(hidden_size, hidden_size, bidirectional=False, batch_first=True)
            for _ in range(3)])
        self.context_gru = nn.GRU(3*hidden_size, hidden_size, 
                bidirectional=False, batch_first=True)
        self.linear = nn.Linear(4*hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, hidden_size)
        self.logvar = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = [g(xi, None) for g, xi in zip(self.NE_grus, x)]
        # context_gru_input = [batch, length, 3*hidden]
        context_gru_input = torch.cat([NE_gru_out[0] for NE_gru_out in x], 2)
        # context_gru_hidden = [1, batch, hidden]
        context_gru_hidden = self.context_gru(context_gru_input, None)[1]
        # cat_hiddens = [1, batch, 4*hidden]
        cat_hiddens = torch.cat((*[NE_gru_out[1] for NE_gru_out in x], context_gru_hidden), 2)
        linear_out = self.linear(cat_hiddens)
        mu = self.mu(linear_out).squeeze()
        logvar = self.logvar(linear_out).squeeze()
        return mu, logvar

class AutoEncoder(nn.Module):
    '''
    Encoder of AutoEncoder.

    Arguments:
    input_size -- 3-tuple for (timing, duration, pitch)'s size
    hidden_size -- hidden_size of grus
    z_size -- dimension of latent vector.
    '''
    def __init__(
            self,
            input_size,
            hidden_size):
        super(AutoEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.NE_grus = nn.ModuleList([
            nn.GRU(hidden_size, hidden_size, bidirectional=False, batch_first=True)
            for _ in range(3)])
        self.context_gru = nn.GRU(3*hidden_size, hidden_size, 
                bidirectional=False, batch_first=True)
        self.linear = nn.Linear(4*hidden_size, int(1.2*hidden_size))
        self.out_linear = nn.Linear(int(1.2*hidden_size), hidden_size)

    def forward(self, x):
        x = [g(xi, None) for g, xi in zip(self.NE_grus, x)]
        # context_gru_input = [batch, length, 3*hidden]
        context_gru_input = torch.cat([NE_gru_out[0] for NE_gru_out in x], 2)
        # context_gru_hidden = [1, batch, hidden]
        context_gru_hidden = self.context_gru(context_gru_input, None)[1]
        # cat_hiddens = [1, batch, 4*hidden]
        cat_hiddens = torch.cat((*[NE_gru_out[1] for NE_gru_out in x], context_gru_hidden), 2)
        linear_out = self.linear(cat_hiddens)
        z = self.out_linear(linear_out).squeeze()
        return z
