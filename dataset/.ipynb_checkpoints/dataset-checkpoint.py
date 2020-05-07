import logging
import os
from datetime import datetime
import pickle as pkl
import random
from multiprocessing import Pool
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

class TBGenomicsDataset(Dataset):
    """ Dataset object used to access the TB-Genomics Dataset """

    def __init__(self, path, shuffle=False, transform=None, subset_size=None):
        """
        Instantiates the Dataset.

        :param root: Path to the folder where the pre-processed dataset is stored.
        :param shuffle: If True, the video files will be shuffled.
        :param transform: Transformations to be done to all frames of the video files.
        """
        self.transform = transform

        self.file = pd.read_csv(path).values.astype(int)
        self.length = self.file.shape[1]                                   
        self.indexes = [idx for idx in range(self.length)]
        (self.p, self.N) = self.hilbert_params(self.file)
        self.hbc = HilbertCurve(self.p, self.N)
        if shuffle:
            random.shuffle(self.indexes)

        if subset_size is not None:
            self.k = int(subset_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        #get the data
        real_idx = self.indexes[idx]
        real = self.file[:, real_idx].reshape(self.file.shape[0], 1)        
        m = np.random.randint(1, self.length, size=(self.k, ))
        while real_idx in m:
            m = np.random.randint(1, self.length, size=(self.k, ))
        

        rest = (self.file[:, m])
        rest = np.concatenate((rest, real), axis=1).T
        rest = pd.DataFrame(rest).values
        
        # temporary params
        C = 1
        H = np.power(2, int(self.p))
        k = rest.shape[0]
        W = np.power(2, int(self.p))

        # convert data from 1D to 2D using hilber transform
        rest = rest.reshape(k, -1)
        data = []
        index = []
        rest_int = rest.astype(int)
        for i in rest_int:
            zero = np.zeros(shape=(H, W))
            coords = []
            for j in i:
                coords.append(self.hbc.coordinates_from_distance(j))
            
            for index, val in enumerate(rest[i]):
                zero[coords[i][0], coords[i][1]] = val

            index.append(coords)
            data.append(zero)

        data = np.array(data)
        data = data.reshape(k, 1, C, H, W)
        data = torch.from_numpy(data)
        data = torch.cat((data, data), dim=1)

        return real_idx, (data, index)
    
    def hilbert_params(self, a):
        """
            Get parameters for Hilbert curve representation
        """ 
        maxv = np.max(a)
        # N = dimensions
        N = 2

        # p = number of iterations
        #
        # 2^(N*p)-1 >= maxv
        # 2^(N*p) >= (maxv + 1)
        # N*p >= log( (maxv + 1), 2 )
        # p >= (1/N) * log( (maxv + 1), 2 )
        p = np.ceil((1/N)*np.log2(maxv+1))
        p = max(8 , p)
        return (p, N) 

