import os
import argparse
import numpy as np

from encoder import *
from utils import traverse_dir

def load_events(path_infile):
    events = []
    with open(path_infile) as f:
        events = [int(token) for token in f.read().split()]
    return events

def compile(path_indir, max_len, ticks_per_beat=1024):
    # list files
    txtfiles = traverse_dir(
        path_indir,
        is_pure=True,
        is_sort=True,
        extension=("txt"))
    n_files = len(txtfiles)
    print('num files:', n_files)

    # Get pad token
    events_index = Event.build_events_index(ticks_per_beat * 4, ticks_per_beat // 4)
    pad_token = Event(events_index, event_type='control', value=3).to_int()

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
                sequence += [pad_token] * (max_len - len(sequence))

            dataset.append(sequence)

    dataset = np.vstack(dataset)
    return dataset

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
    train_data = compile(args.path_train_indir, args.max_len)
    test_data = compile(args.path_test_indir, args.max_len)

    print('---')
    print(' > train x:', train_data.shape)
    print(' >  test x:', test_data.shape)

    # Save datasets
    path_train_outfile = os.path.join(args.path_outdir, 'train.npz')
    path_test_outfile = os.path.join(args.path_outdir, 'test.npz')

    np.savez(path_train_outfile, x=train_data)
    np.savez(path_test_outfile, x=test_data)
