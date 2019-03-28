import torch
import torch.nn as nn
import torch.nn.functional as F

class NoteUnrollingDecoder(nn.Module):
    '''
    NoteUnrollingDecoder.
    Decode latent vectors to notes.

    Arguments:
    hisdden_size -- hidden_size latent vector
    output_size -- 3-tuple for (timing, duration, pitch)'s size
    '''
    def __init__(
            self,
            hidden_size,
            output_size):
        super(NoteUnrollingDecoder, self).__init__()
        self.gru1 = nn.ModuleList(
                [nn.GRUCell(hidden_size * 2, hidden_size)
                    for _ in range(3)])
        self.context = nn.GRUCell(hidden_size * 3, hidden_size)
        self.gru2 = nn.ModuleList(
                [nn.GRUCell(hidden_size * 2, hidden_size)
                    for _ in range(3)])
        self.outputs = nn.ModuleList(
                [nn.Linear(hidden_size, o) for o in output_size])
        self.hidden = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, z, embeds, length=100,
            teacher_forcing=False, target=None):
        
        batch_size = z.size(0)

        hidden = F.tanh(self.hidden(z))
        gru1_hidden = [hidden for _ in range(3)]
        gru2_hidden = [hidden for _ in range(3)]
        context = hidden

        outputs = []
        note = [torch.zeros(batch_size, dtype=torch.long) for _ in range(3)]
        
        if next(self.parameters()).is_cuda:
            note = [n.cuda() for n in note]
        
        note = [e(x) for e, x in zip(embeds, note)]

        for t in range(length):
            gru1_hidden = [
                    g(torch.cat([x, z], -1), h)
                    for g, x, h in zip(self.gru1, note, gru1_hidden)]
            gru1_hidden_unroll = [h for h in gru1_hidden]
            output = []

            # Note Unrolling
            for i in range(3):
                context = self.context(torch.cat(gru1_hidden_unroll, -1), context)
                gru2_hidden[i] = self.gru2[i](
                        torch.cat([context, z], -1),
                        gru2_hidden[i])                
                output.append(self.outputs[i](gru1_hidden_unroll[i] + gru2_hidden[i]))
                if teacher_forcing:
                    note[i] = embeds[i](target[i, :, t])
                else:
                    note[i] = embeds[i](output[i].max(-1)[-1].detach())
                gru1_hidden_unroll[i] = self.gru1[1](
                    torch.cat([note[i], z], -1),
                    gru1_hidden_unroll[i])
            
            outputs.append(output)
        outputs = [torch.stack(o).transpose(0, 1) for o in zip(*outputs)]
        return outputs


class NoteEventDecoder(nn.Module):
    '''
    kaikai add
    Decoder Without Note Unrolling.
    Decode latent vectors to notes.

    Arguments:
    hisdden_size -- hidden_size latent vector
    output_size -- 3-tuple for (timing, duration, pitch)'s size
    '''
    def __init__(
            self,
            hidden_size,
            output_size):
        super(NoteEventDecoder, self).__init__()
        self.gru1 = nn.ModuleList(
                [nn.GRUCell(hidden_size * 2, hidden_size)
                    for _ in range(3)])
        self.context = nn.GRUCell(hidden_size * 3, hidden_size)
        self.gru2 = nn.ModuleList(
                [nn.GRUCell(hidden_size * 2, hidden_size)
                    for _ in range(3)])
        self.outputs = nn.ModuleList(
                [nn.Linear(hidden_size, o) for o in output_size])
        self.hidden = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, z, embeds, length=100,
            teacher_forcing=False, target=None):
        
        batch_size = z.size(0)

        hidden = F.tanh(self.hidden(z))
        gru1_hidden = [hidden for _ in range(3)]
        gru2_hidden = [hidden for _ in range(3)]
        context = hidden

        outputs = []
        note = [torch.zeros(batch_size, dtype=torch.long) for _ in range(3)]
        
        if next(self.parameters()).is_cuda:
            note = [n.cuda() for n in note]
        
        note = [e(x) for e, x in zip(embeds, note)]

        for t in range(length):
            gru1_hidden = [
                    g(torch.cat([x, z], -1), h)
                    for g, x, h in zip(self.gru1, note, gru1_hidden)]
            gru1_hidden_unroll = [h for h in gru1_hidden]
            output = []

            # Without Note Unrolling
            context = self.context(torch.cat(gru1_hidden_unroll, -1), context)
            for i in range(3):
                gru2_hidden[i] = self.gru2[i](
                        torch.cat([context, z], -1),
                        gru2_hidden[i])                
                output.append(self.outputs[i](gru1_hidden_unroll[i] + gru2_hidden[i]))
                if teacher_forcing:
                    note[i] = embeds[i](target[i, :, t])
                else:
                    note[i] = embeds[i](output[i].max(-1)[-1].detach())
            
            outputs.append(output)
        outputs = [torch.stack(o).transpose(0, 1) for o in zip(*outputs)]
        return outputs


