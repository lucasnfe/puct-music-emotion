import os
import torch
import csv
import json
import numpy as np

from sklearn.model_selection import GroupKFold

class VGMidiLabelled(torch.utils.data.Dataset):
    def __init__(self, path_data):
        data = np.load(path_data)
        self.pieces = torch.from_numpy(data['x']).long()
        self.labels = torch.from_numpy(data['y']).float()
        
    def __getitem__(self, idx):
        x = self.pieces[idx]
        y = self.labels[idx]
        return x,y

    def __len__(self):
        return len(self.pieces)
