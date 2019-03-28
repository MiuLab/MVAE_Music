import pickle
import argparse
import os

from midi_utils import read_midi_files
from model import ModularizedVAE
from loader import MusicDataset, BatchGenerator
from args import get_args
import scheduler

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from progressbar import ProgressBar, ETA, FormatLabel, Bar

class DataPathOrderNotRightError(Exception):
    pass

def assert_data_order(data_path):
    # Assert 
    if ("nottingham" not in data_path[0].lower()) or ("jsb" not in data_path[1].lower()) or ("piano" not in data_path[2].lower()):
        print("First argument should be nottingham, second should be jsb and third should be piano")
        raise DataPathOrderNotRightError("You should place data path in the right order")

if __name__ == '__main__':
    
    args = get_args()

    print('Loading data......')
    # Assert data path's order
    assert_data_order(args.data)
    # Load
    datass, _, note_sets = read_midi_files(args.data, None)
    
    train_loader = BatchGenerator(datass, batch_size=args.batch_size, shuffle=True)
    
    print('Build model......')
    model_args = {
            'note_size': (
                len(note_sets['timing']),
                len(note_sets['duration']),
                len(note_sets['pitch'])),
            'hidden_size': args.hidden_size,
            'encoder_name': args.encoder,
            'decoder_name': args.decoder
            }
    vae = ModularizedVAE(**model_args)

    if args.cuda:
        vae.cuda()

    optimizer = torch.optim.RMSprop(
            vae.parameters(),
            lr=args.lr,
            alpha=args.alpha)
    
    scheduled_sampler = scheduler.LogisticScheduler(0.0, 0.01, 0.9, 10)

    log_file = open(os.path.join('logs', args.prefix + '.log'), 'w+')
    print('epoch,rec_loss,kld_loss,accuracy,schedule_rate', file=log_file)
    log_file.flush()

    for epoch in range(1, args.epochs + 1):

        print('Epoch:', epoch)
        total_loss = []
        total_kld = []
        total_acc = []
        
        widgets = [FormatLabel(''), ' ',
                Bar('=', '[', ']'), ' - ',
                ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=len(train_loader))
        pbar.start()

        for batch_i, batch_song in enumerate(train_loader()):

            batch_size = batch_song.size(0)
            batch_song = batch_song.permute(2, 0, 1)
            if args.cuda:
                batch_song = batch_song.cuda()
            
            if args.encoder == "AutoEncoder":
                reconstruct, song, z = vae(
                        batch_song,
                        teacher_forcing=True)
            else:
                reconstruct, song, (mu, logvar) = vae(
                        batch_song,
                        teacher_forcing=True)
            rec_loss = vae.reconstruct_loss(reconstruct, batch_song)
            if args.encoder == "AutoEncoder":
                # autoencoder do not have kld loss
                kld_loss = torch.zeros(1).cuda() if args.cuda else torch.zeros(1)
            else:
                kld_loss = vae.kld_loss(mu, logvar)
            loss = rec_loss + \
                    scheduled_sampler.get_value(epoch) * kld_loss
            total_loss.append(loss.item())
            total_kld.append(kld_loss.item())
            total_acc.append((song == batch_song).float().mean().item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            widgets[0] = FormatLabel('{}/{}'.format(
                batch_i * args.batch_size + batch_size, len(train_loader)))
            pbar.update(batch_i * args.batch_size + batch_size)
            
        pbar.finish()

        print('rec_loss: {:.4f}, kld_loss: {:.4f}, accuracy: {:.4f}, shedule_rate: {:.4f}'.format(
            np.mean(total_loss),
            np.mean(total_kld),
            np.mean(total_acc),
            scheduled_sampler.get_value(epoch)))
        
        # Train log
        print(epoch, np.mean(total_loss), np.mean(total_kld), np.mean(total_acc),
            scheduled_sampler.get_value(epoch), sep=',', file=log_file)
        log_file.flush()
        #
        if epoch % args.save_intervals == 0:
            save_model_dir = os.path.join('models', args.prefix)
            os.mkdir(save_model_dir) if os.path.isdir(save_model_dir) == False else None
            torch.save({
                    'state_dict': {k: v.cpu() for k, v in vae.state_dict().items()},
                    'model_args': model_args,
                    'note_sets': note_sets},
                open(os.path.join(save_model_dir,
                    '%s_e%d.pt' % (args.prefix, epoch)), 'wb+'))

