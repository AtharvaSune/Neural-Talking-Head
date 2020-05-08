import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import os

class VoxCelebDataset(Dataset):
    def __init__(self, root, transform, shuffle=True):
        self.root = root
        self.transform = transform
        self.files = os.listdir(root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pass