import os
import torch
import csv
import json
import numpy as np

from sklearn.model_selection import GroupKFold

PAD_TOKEN = 390

def pad_collate(batch):
    max_len = max([example[0].shape[-1] for example in batch])

    padded_examples = []
    targets = []

    for example in batch:
        x, y = example

        padding_len = max_len - x.shape[-1]
        if padding_len > 0:
            padding = torch.full((padding_len,), PAD_TOKEN, dtype=torch.int64)
            x = torch.cat((x, padding), dim=0)

        padded_examples.append(x)
        targets.append(y)

    return torch.stack(padded_examples), torch.stack(targets)

class VGMidiLabelled(torch.utils.data.Dataset):
    def __init__(self, midi_csv, seq_len, balance=False, prefix=0):
        self.seq_len = seq_len
        pieces, labels, groups = self._load_pieces(midi_csv, seq_len, prefix)

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        return torch.tensor(self.pieces[idx]), torch.tensor(self.labels[idx])

    def _load_txt(self, file_path):
        loaded_list = []
        with open(file_path) as f:
            loaded_list = [int(token) for token in f.read().split()]
        return loaded_list

    def _load_pieces(self, midi_csv, seq_len, prefix_step=0):
        pieces = []
        labels = []
        groups = []

        csv_dir, csv_name = os.path.split(midi_csv)
        for row in csv.DictReader(open(midi_csv, "r")):
            piece_path = os.path.join(csv_dir, row["midi"])

            # Make sure the piece has been encoded into a txt file (see encoder.py)
            file_name, extension = os.path.splitext(piece_path)
            if os.path.isfile(file_name + ".txt"):
                # Load entire piece
                encoded = self._load_txt(file_name + ".txt")

                # Trim encoded piece to max len
                encoded = encoded[:seq_len]

                # Get emotion
                emotion = VGMidiEmotion(int(row["valence"]), int(row["arousal"]))

                # Add piece
                pieces.append(encoded)
                labels.append(emotion.get_quadrant())
                groups.append(row["game"])

                if prefix_step > 0:
                    # Generate prefixes of incresing size
                    for prefix_size in range(prefix_step, len(encoded), prefix_step):
                        pieces.append(encoded[:prefix_size])
                        labels.append(emotion.get_quadrant())
                        groups.append(row["game"])

        assert len(pieces) == len(labels) == len(groups)
        return pieces, labels, groups

    def get_pieces_txt(self):
        pieces = []
        for piece in self.pieces:
            piece_txt = " ".join([str(token) for token in piece])
            pieces.append(piece_txt)

        return pieces

class VGMidiEmotion:
    def __init__(self, valence, arousal):
        assert valence in (-1, 1)
        assert arousal in (-1, 1)

        self.va = np.array([valence, arousal])
        self.quad2emotion = {0: "q1", 1: "q2", 2: "q3", 3: "q4"}

    def __eq__(self, other):
        if other == None:
            return False
        return (self.va == other.va).all()

    def __ne__(self, other):
        if other == None:
            return True
        return (self.va != other.va).any()

    def __str__(self):
        return self.quad2emotion[self.get_quadrant()]

    def __getitem__(self, key):
        return self.va[key]

    def __setitem__(self, key, value):
        self.va[key] = value

    def get_quadrant(self):
        if self.va[0] == 1 and self.va[1] == 1:
            return 0
        elif self.va[0] == -1 and self.va[1] == 1:
            return 1
        elif self.va[0] == -1 and self.va[1] == -1:
            return 2
        elif self.va[0] == 1 and self.va[1] == -1:
            return 3

        return None

class VGMidiSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, bucket_size=64, max_len=2048, shuffle=False):
        self.max_len = max_len
        self.bucket_size = bucket_size
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        buckets = [[] for i in range(0, self.max_len, self.bucket_size)]

        for i, example in enumerate(self.data_source):
            bucket_ix = len(example) // self.bucket_size - 1
            buckets[bucket_ix].append(i)

        idxs = []
        for i in range(len(buckets)):
            if self.shuffle:
                np.random.shuffle(buckets[i])
            idxs += buckets[i]

        return iter(idxs)

    def __len__(self):
        return len(self.data_source)
