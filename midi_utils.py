from IPython import embed
import os
from pretty_midi import PrettyMIDI, Instrument, Note
import numpy as np

MAX_DURATION = 256

def read_one_midi_and_cut(filename, output, cut_length=16):
    data = PrettyMIDI(filename)
    notes = []
    for track in data.instruments:
        notes += track.notes
    notes = sorted(notes, key=lambda note: note.start)
    #print(notes)
    end_time = 0
    for note in reversed(notes):
        if note.end > end_time:
            end_time = note.end
    
    start_time = 0
    end_time = max(0, end_time - cut_length)

    import random
    cut_start_time = random.uniform(start_time, end_time)
    cut_end_time = cut_start_time + cut_length

    print('%s: start=%.1f, end=%.1f' % (filename, cut_start_time, cut_end_time))

    midi_length = len(notes)
    for i in reversed(range(midi_length)):
        if notes[i].start < cut_start_time or\
                notes[i].start > cut_end_time:
            del notes[i]
        elif notes[i].end > cut_end_time:
            notes[i].end = cut_end_time
    for note in notes:
        note.start -= cut_start_time
        note.end -= cut_start_time

    midi_file = PrettyMIDI(resolution=220, initial_tempo=120)
    track = Instrument(0)
    track.notes = notes
    midi_file.instruments.append(track)
    midi_file.write(output)
    
def cut_midis(midi_dir, save_dir):
    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)
    dirs = os.listdir(midi_dir)
    for one_dir in dirs:
        if os.path.isdir(os.path.join(midi_dir, one_dir)) == False:
            continue
        if os.path.isdir(os.path.join(save_dir, one_dir)) == False:
            os.mkdir(os.path.join(save_dir, one_dir))
        files = os.listdir(os.path.join(midi_dir, one_dir))
        for filename in files:
            full_filename = os.path.join(midi_dir, one_dir, filename)
            save_filename = os.path.join(save_dir, one_dir, filename)
            read_one_midi_and_cut(full_filename, save_filename)

def encode_midi(midi_file, note_sets):
    if len(midi_file.time_signature_changes) > 1:
        return None, None
    #if len(midi_file.get_tempo_changes()[0]) > 1:
    #    return None, None

    start = 0
    notes = []
    for track in midi_file.instruments:
        notes += track.notes
    notes = sorted(notes, key=lambda note : note.start) 
    data = []
    change_time, tempi = midi_file.get_tempo_changes()
    i = 0
    for note in notes:
        if i + 1 < len(change_time) and note.start >= change_time[i + 1]:
            i  = i + 1
        # Quantize to 16th note(That is why we need to multiply 4 here)
        timing = int(round((note.start - start) / 60 * tempi[i] * 4))
        duration = int(round((note.end - note.start) / 60 * tempi[i] * 4))
        pitch = note.pitch
        start = note.start
        data.append([timing, duration, pitch])
        note_sets['timing'].add(timing) 
        note_sets['duration'].add(duration)
        note_sets['pitch'].add(pitch)
    return np.array(data), midi_file.estimate_tempo()

def pitch_augmentation(data, note_sets, shift=range(-3, 4)):
    datas = []
    for s in shift:
        d = data.copy()
        assert all(np.less_equal(0, d[:, -1] + s)) and all(np.less_equal(d[:, -1] + s, 127)), "Pitch augment caused out-of-range pitch"
        d[:, -1] += s
        note_sets['pitch'] |= set(d[:, -1])
        datas.append(d)
    return datas

def speed_augmentation(data, note_sets):
    datas = []

    # 2x speed
    if (data[:, 0:1] % 2).sum() == 0:
        d = data.copy()
        d[:, 0:1] //= 2
        note_sets['timing'] |= set(d[:, 0])
        note_sets['duration'] |= set(d[:, 1])
        datas.append(d)
    else:
        datas.append(data)
    '''
    # 0.5x speed
    if data[:, 0:1].max() < 16:
        d = data.copy()
        d[:, 0:1] *= 2
        note_sets['timing'] |= set(d[:, 0])
        note_sets['duration'] |= set(d[:, 1])
        datas.append(d)
    '''
    return datas

def shift_augmentation(data, max_length, stride):
    datas = []
    if len(data) < max_length:
        return datas
    for i in range(0, len(data), stride):
        datas.append(data[i : i + max_length].copy())
        if len(datas[-1]) < max_length:
            datas[-1] = data[-max_length:].copy()
            break
    return datas

def read_midi_files(paths, valid_paths):
    '''
        Input: a list of path, valid_paths(if valid path exist)
        Output: datas, note_sets
    '''
    # Assert a path
    assert isinstance(paths, list)
    datass = [] # a list of datas = (N, T, 3)
    valid_datass = [] # a list of validation datass
    note_sets = {
            'timing': set(),
            'duration': set(),
            'pitch': set()}

    def read_midi_files_with_note_sets(paths, datass, note_sets):
        for path in paths:
            datas = []
            for filename in os.listdir(path):
                if filename.endswith('mid'):
                    midi_file = PrettyMIDI(os.path.join(path, filename))
                    data, tempo = encode_midi(midi_file, note_sets)
                    if not isinstance(data, np.ndarray):
                        continue

                    data = pitch_augmentation(data, note_sets)  
                    
                    for d in data:
                        datas += shift_augmentation(d, 100, 50)
                    
                        '''
                        shifted_data = shift_augmentation(d, 100, 50)
                        if tempo < 180:
                            datas += shifted_data
                        else:
                            for sd in shifted_data:
                                datas += speed_augmentation(sd, note_sets)
                        '''
            # Append datas
            datass.append(datas)
    # train set
    read_midi_files_with_note_sets(paths, datass, note_sets)
    # valid set
    if valid_paths is not None:
        read_midi_files_with_note_sets(valid_paths, valid_datass, note_sets)
    # 
    note_sets = {k: ['<padding>'] + list(v) for k, v in note_sets.items()}
    note_dicts = {
            key: {x: i for i, x in enumerate(value)}
            for key, value in note_sets.items()}
    # NOTE: datass will be changed inplace
    def map_value_to_index(datass):
        # Map each real value to index
        for k in range(len(datass)): # Loop over different folders
            for i in range(len(datass[k])): # Loop over N
                for j, note in enumerate(datass[k][i]): # Loop over T
                    datass[k][i][j, 0] = note_dicts['timing'][note[0]]
                    datass[k][i][j, 1] = note_dicts['duration'][note[1]]
                    datass[k][i][j, 2] = note_dicts['pitch'][note[2]]
            datass[k] = np.array(datass[k]).astype(np.int64) # Stack a folder's midi example into (N, T, 3)
    # train set
    map_value_to_index(datass)
    # valid set
    if len(valid_datass) != 0:
        map_value_to_index(valid_datass)
    return datass, valid_datass, note_sets

def dump_midi(data, note_sets, path):
    midi_file = PrettyMIDI(resolution=220, initial_tempo=120)
    track = Instrument(0)
    time = 0

    # Shift first timing to 0
    #time -= note_sets['timing'][data[0][0]] * 30
    
    for note in data:
        # <padding> == 0
        if note[0] == 0:
            continue
        time += note_sets['timing'][note[0]] * 15 / 120
        track.notes.append(Note(
            velocity=100,
            start=time,
            end=time + note_sets['duration'][note[1]] * 15 / 120,
            pitch=note_sets['pitch'][note[2]]))
        #print(track.notes[-1])
    midi_file.instruments.append(track)
    midi_file.write(path)



if __name__ == '__main__':
    #data, note_sets = read_midi_files('/tmp2/andy920262/jsb')
    #data, note_sets = read_midi_files('/tmp2/andy920262/piano-midi.de')
    #data, note_sets = read_midi_files('/tmp2/andy920262/nottingham/midi')
    #dump_midi(data[-1], note_sets, 'output.mid')

    cut_midis('random_sampled', 'cut_random_sampled')   
    #read_one_midi_and_cut('midis/fe_nu_e130_0.mid', 'output.mid')
