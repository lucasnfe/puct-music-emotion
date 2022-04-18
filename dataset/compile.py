import os
import argparse
import json
import numpy as np

from encoder import *
from utils import traverse_dir

def load_events(path_infile):
    events = []
    with open(path_infile) as f:
        events = f.read().split()
    return events

def build_vocabulary(train_data, test_data):
    vocabulary = set()

    for sentence in train_data:
        vocabulary = vocabulary | set(sentence)

    for sentence in test_data:
        vocabulary = vocabulary | set(sentence)

    # sort vocabulary
    vocabulary = sorted(list(vocabulary))
    return {vocabulary[i]:i for i in range(len(vocabulary))}

def load_dataset(path_indir, max_len, ticks_per_beat=1024):
    # list files
    txtfiles = traverse_dir(
        path_indir,
        is_pure=True,
        is_sort=True,
        extension=("txt"))
    n_files = len(txtfiles)
    print('num files:', n_files)

    # Make room for split x[:-1], y[1:]
    max_len += 1

    dataset = []
    for fidx in range(n_files):
        path_txt = txtfiles[fidx]
        print('{}/{}'.format(fidx, path_txt))

        # Load events
        path_infile = os.path.join(path_indir, path_txt)
        events = load_events(path_infile)

        # Split the piece into sequences of len seq_len
        for i in range(0, len(events), max_len):
            sequence = events[i:i+max_len]
            if len(sequence) < max_len:
                # Pad sequence
                sequence += ['.'] * (max_len - len(sequence))

            dataset.append(sequence)

    return dataset

def compile(dataset, vocabulary):
    compiled_data = []
    for sequence in dataset:
        # Encode sequence of events (str) using the vocabulary
        encoded_sequence = []
        for event in sequence:
            encoded_sequence.append(vocabulary[event])

        compiled_data.append(encoded_sequence)

    return np.vstack(compiled_data)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='compile.py')
    parser.add_argument('--path_train_indir', type=str, required=True)
    parser.add_argument('--path_test_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    parser.add_argument('--max_len', type=int, required=True)
    args = parser.parse_args()

    os.makedirs(args.path_outdir, exist_ok=True)

    # Load datasets
    train_data = load_dataset(args.path_train_indir, args.max_len)
    test_data = load_dataset(args.path_test_indir, args.max_len)

    # Build vocabulary
    vocabulary = build_vocabulary(train_data, test_data)

    # Compile data
    train_data = compile(train_data, vocabulary)
    test_data = compile(test_data, vocabulary)

    print('---')
    print(' > train x:', train_data.shape)
    print(' >  test x:', test_data.shape)
    print(' > vocabulary:')
    print(vocabulary)

    # Save datasets
    path_train_outfile = os.path.join(args.path_outdir, 'train.npz')
    path_test_outfile = os.path.join(args.path_outdir, 'test.npz')

    np.savez(path_train_outfile, x=train_data)
    np.savez(path_test_outfile, x=test_data)

    # Save vocabulary
    path_vocab_outfile = os.path.join(args.path_outdir, 'vocabulary.json')
    with open(path_vocab_outfile, "w") as f:
        json.dump(vocabulary, f)
