import torch
import torch.nn as nn
import torch.nn.functional as F
from model_architectures import encoder
from model_architectures import decoder
from model_architectures import vae


class ModularizedVAE(vae.BaseVAE):
    def __init__(
            self,
            note_size,
            hidden_size,
            encoder_name='NoteEventEncoder',
            decoder_name='NoteUnrollingDecoder',
            autoencoder=False):
        try:
                super(ModularizedVAE, self).__init__(
                        getattr(encoder, encoder_name)(
                            input_size=hidden_size,
                            hidden_size=hidden_size),
                        getattr(decoder, decoder_name)(
                            hidden_size=hidden_size,
                            output_size=note_size))
        except AttributeError:
            print('Encoder or decoder name not found: %s, %s' % \
                (encoder_name, decoder_name))
            exit(0)

        self.autoencoder = True if encoder_name == "AutoEncoder" else False
        self.embeddings = nn.ModuleList(
                [nn.Embedding(i, hidden_size) for i in note_size])
        self.hidden_size = hidden_size
    
    def encode(self, x):
        x = [e(xi) for e, xi in zip(self.embeddings, x)]
        return self._encoder(x)

    def decode(self, x, length=100, teacher_forcing=False, truth=None):
        return self._decoder(x, self.embeddings, length, teacher_forcing, truth)
    
    def sample(self, length=100):
        z = torch.zeros(1, self.hidden_size).normal_(0, 1)
        if next(self.parameters()).is_cuda:
            z = z.cuda()
        output = self.decode(z, length)
        output = [o.max(-1)[-1] for o in output]
        output = torch.stack(output).squeeze().transpose(0, 1)
        return output.detach().cpu().numpy()

    def reconstruct_loss(self, predict, target):
        '''
            predict is a list contain three (batch_size, timestep, note) tensor
        '''
        timing_voc_size = predict[0].size()[-1]
        duration_voc_size = predict[1].size()[-1]
        pitch_voc_size = predict[2].size()[-1]
        loss = 0
        for p, t in zip(predict, target):
            for i in range(p.size(1)):
                loss += F.cross_entropy(p[:, i], t[:, i])
                #loss += F.cross_entropy(p[:, i], t[:, i], ignore_index=0)
        return loss
    
    def forward(self, x, teacher_forcing=True):
        length = x[0].size(1)
        if self.autoencoder:
            z = self.encode(x)
        else:
            mu, logvar = self.encode(x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        reconstruct = self.decode(z, length, teacher_forcing, x)
        song = [o.max(-1)[-1] for o in reconstruct]
        song = torch.stack(song).detach()
        if self.autoencoder:
            return reconstruct, song, z
        else:
            return reconstruct, song, (mu, logvar)

