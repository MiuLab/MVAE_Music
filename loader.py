import numpy as np
import random
import copy
import itertools
import torch
from torch.utils.data import Dataset
import math

class MusicDataset(Dataset):
    def __init__(self, datass):
        self.data = np.concatenate(datass, axis=0)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
    
class BatchGenerator():
    def __init__(self, datas, batch_size, shuffle=True):
        self.datas = copy.deepcopy(datas)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ratio = [0.3, 0.2, 0.5] # ratio of nottingham, jdb, piano
        print(self.ratio)
        # Get max length
        max_length = len(max(datas, key=len))
        # Compute batch information
        number_of_batch = max_length // batch_size
        if max_length % batch_size != 0:
            number_of_batch += 1
        self.number_of_batch = number_of_batch 

    def __call__(self):
        '''
            WARNING: datas will be modified
            Output: a data balanced batch
            WARNING:
                ratios correspond to 1. nottingham, 2. jsb, 3. piano-midi
        '''
        ratios = self.ratio
        datas = self.datas
        # Shuffle all the datas[i]
        for i in range(len(datas)): # Loop over different folders
            # Create indices
            shuffle_indices = list(range(len(datas[i]))) # Shuffle each folder's midi examples
            random.shuffle(shuffle_indices)
            # Index that indices
            datas[i] = datas[i][shuffle_indices]
        # Create infinite cycle for each folder's midi example by itertools
        cycles = [itertools.cycle(shuffle_data) for shuffle_data in datas]
        cycles_batch_size = []
        if ratios is None:
            for i in range(len(cycles)):
                if i != 0:
                    cycles_batch_size.append(self.batch_size // len(cycles))
                else:
                    cycles_batch_size.append(self.batch_size // len(cycles) + self.batch_size % len(cycles))
        else:
            for i in range(len(cycles)):
                cycles_batch_size.append(math.floor(self.batch_size * ratios[i]))
            # 
            remainder = self.batch_size - sum(cycles_batch_size)
            if remainder > 0:
                cycles_batch_size[0] += remainder

        # Create batch
        for i in range(self.number_of_batch): # Loop over batch
            batch = []
            for cycle, cycle_batch_size in zip(cycles, cycles_batch_size):
                # Get `batch_size` number of samples
                cycle_batch = [next(cycle) for _ in range(cycle_batch_size)]
                # extend cycle batch
                batch.extend(cycle_batch)    
            # transform to numpy
            batch = np.array(batch)
            # yield a batch
            yield torch.from_numpy(batch).long()
            del batch

    def __len__(self):
        return self.number_of_batch * self.batch_size
    

        
        
        
