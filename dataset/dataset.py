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

def get_landmarks(data):
    """
    Function to return the indices of genes that are potential landmarks
    """
    data = data.T
    idx = []
    print
    for i in range(data.shape[1]):
        tmp = data[:, i] != 0
        if np.sum(tmp) == 0:
            idx.append(i)
    if (len(idx) == 0):
        for i in range(data.shape[1]):
            idx.append(i)
    return idx


class TBGenomicsDataset(Dataset):
    """ Dataset object used to access the TB-Genomics Dataset """

    def __init__(self, path, shuffle=False, subset_size=None):
        """
        Instantiates the Dataset.

        :param root: Path to the folder where the pre-processed dataset is stored.
        :param shuffle: If True, the video files will be shuffled.
        """

        # read the file and generate nescessary values
        self.file = pd.read_csv(path).values
        self.file[self.file<=0] = 0
        self.length = self.file.shape[1]                                   
        self.indexes = [idx for idx in range(self.length)]
        (self.p, self.N) = self.hilbert_params(self.file)
        self.hbc = HilbertCurve(self.p, self.N)
        self.landmarks_idx = get_landmarks(self.file)
        print("H", len(self.landmarks_idx))

        self.file = self.file.T
        if shuffle:
            random.shuffle(self.indexes)

        if subset_size is not None:
            self.k = int(subset_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        #get the data for idx
        real_idx = self.indexes[idx]
        real = self.file[real_idx, :].reshape(self.file.shape[1], 1)
        landmarks_real = self.file[real_idx ,self.landmarks_idx].reshape(self.file.shape[1], 1)
        
        # the remaining k
        m = np.random.randint(1, self.length, size=(self.k, ))
        while real_idx in m:
            m = np.random.randint(1, self.length, size=(self.k, ))
        rest = (self.file[m, :]).reshape(self.file.shape[1], -1)
        landmarks_k = self.file[m, :]
        landmarks_k = landmarks_k[:, self.landmarks_idx].reshape(self.file.shape[1], -1)

        # concatenate to get combined form
        rest = np.concatenate((rest, real), axis=1).T
        rest = pd.DataFrame(rest).values

        landmarks = np.concatenate((landmarks_k, landmarks_real), axis=1).T
        landmarks = pd.DataFrame(landmarks).values

        # temporary params
        C = 1
        H = np.power(2, int(self.p))
        k = rest.shape[0]
        W = np.power(2, int(self.p))

        # convert data from 1D to 2D using hilber transform
        rest = rest.reshape(k, -1)
        data = []
        landmarks_final = []
        rest_int = rest.astype(int)
        landmarks_int = landmarks.astype(int)

        # convert values to 2d using hilbert transform
        for i, val in enumerate(rest_int):
            zero = np.zeros(shape=(H, W))
            coords = []
            for j in val:
                coords.append(self.hbc.coordinates_from_distance(j))
            for index, val in enumerate(rest[i]):
                zero[coords[index][0], coords[index][1]] = val

            data.append(zero)
        
        data = np.array(data)
        data = data.reshape(k, 1, C, H, W)
        data = torch.from_numpy(data)

        # convert landmarks to 2d using hilbert transform
        for i, val in enumerate(landmarks_int):
            zero = np.zeros(shape=(H, W))
            coords = []
            for j in val:
                coords.append(self.hbc.coordinates_from_distance(j))
            
            for index, val in enumerate(landmarks[i]):
                zero[coords[index][0], coords[index][1]] = val

            landmarks_final.append(zero)
        
        landmarks = np.array(landmarks_final)
        landmarks = landmarks.reshape(k, 1, C, H, W)
        landmarks = torch.from_numpy(landmarks)        

        data = torch.cat((data, landmarks), dim=1)

        return real_idx, data
    
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
