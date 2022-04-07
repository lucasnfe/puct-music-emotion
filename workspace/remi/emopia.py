import torch
import numpy as np

class Emopia(torch.utils.data.Dataset):
    def __init__(self, path_data):
        data = np.load(path_data)
        self.x = torch.from_numpy(data['x']).long()
        self.y = torch.from_numpy(data['y']).long()
        self.mask = torch.from_numpy(data['mask']).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.mask[index]

    def __len__(self):
        return len(self.x)
