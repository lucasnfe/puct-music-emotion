import os
import torch
import csv
import json
import numpy as np

from sklearn.model_selection import GroupKFold

class VGMidiLabelled(torch.utils.data.Dataset):
    def __init__(self, path_data, pad_token):
        data = np.load(path_data)
        self.pieces = torch.from_numpy(data['x']).long()
        self.labels = torch.from_numpy(data['y']).long()
        self.pad_token = pad_token

    def __getitem__(self, idx):
        x = self.pieces[idx]
        y = self.labels[idx]
        lengths = (x != self.pad_token).sum(dim=-1)
        return x, y, lengths

    def __len__(self):
        return len(self.pieces)
