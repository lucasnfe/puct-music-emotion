import os
import torch
import csv
import json
import numpy as np

from sklearn.model_selection import GroupKFold

class VGMidiLabelled(torch.utils.data.Dataset):
    def __init__(self, path_data, pad_token, bar_token, generate_prefixes):
        data = np.load(path_data)
        self.pad_token = pad_token
        self.bar_token = bar_token
        
        if generate_prefixes:
            self.pieces, self.labels = self._generate_prefixes(data['x'], data['y'], bar_token)
        else:
            self.pieces = torch.from_numpy(data['x']).long()
            self.labels = torch.from_numpy(data['y']).long()
        
    def __getitem__(self, idx):
        x = self.pieces[idx]
        y = self.labels[idx] - 1
        
        lengths = (x != self.pad_token).sum(dim=-1)
        return x, y, lengths

    def __len__(self):
        return len(self.pieces)

    def _generate_prefixes(self, pieces, labels, bar_token):
        prefixes_x = []
        prefixes_y = []

        for piece, label in zip(pieces, labels):
            # Generate prefixes of incresing size
            prefix = []
            n_bars = 0
            for idx in piece:
                prefix.append(idx)
                if idx == bar_token:
                    n_bars += 1
                    if n_bars % 1 == 0:
                        prefixes_x.append(prefix + [self.pad_token] * (piece.shape[-1] - len(prefix)))
                        prefixes_y.append(label)

            prefixes_x.append(prefix)
            prefixes_y.append(label)

        prefixes_x = np.vstack(prefixes_x)
        prefixes_y = np.vstack(prefixes_y)

        return torch.from_numpy(prefixes_x), torch.from_numpy(prefixes_y)
        
