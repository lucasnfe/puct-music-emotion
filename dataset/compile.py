import os
import argparse
import numpy as np

from encoder import *
from utils import traverse_dir

def load_events_idx(path_infile):
    events_idx = []
    with open(path_infile) as f:
        events_idx = [int(idx) for idx in f.read().split()]
    return events_idx

def load_emotion(events_idx):
    emotions = []
    for idx in events_idx:
        event = Event.from_int(idx)
        if event.type == 'emotion':
            emotions.append(event.value)

    # Make sure there's only one emotion event in a piece
    assert len(emotions) == 1
    return emotions[0]

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
    pad_token = Event(event_type='control', value=3).to_int()

    pieces = []
    labels = []
    for fidx in range(n_files):
        path_txt = txtfiles[fidx]
        print('{}/{}'.format(fidx, path_txt))

        # Load events
        path_infile = os.path.join(path_indir, path_txt)
        events_idx = load_events_idx(path_infile)

        # Load emotion
        emotion = load_emotion(events_idx)

        # Split the piece into sequences of len seq_len
        for i in range(0, len(events_idx), max_len):
            sequence = events_idx[i:i+max_len]
            if len(sequence) < max_len:
                # Pad sequence
                sequence += [pad_token] * (max_len - len(sequence))

            pieces.append(sequence)
            labels.append(emotion)

    pieces = np.vstack(pieces)
    labels = np.vstack(labels)

    assert pieces.shape[0] == labels.shape[0]
    return pieces, labels

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
    train_pieces, train_labels = compile(args.path_train_indir, args.max_len)
    test_pieces, test_labels = compile(args.path_test_indir, args.max_len)

    print('---')
    print(' > train x:', train_pieces.shape)
    print(' > train y:', train_labels.shape)
    print(' >  test x:', test_pieces.shape)
    print(' >  test y:', test_labels.shape)

    # Save datasets
    path_train_outfile = os.path.join(args.path_outdir, 'train.npz')
    path_test_outfile = os.path.join(args.path_outdir, 'test.npz')

    np.savez(path_train_outfile, x=train_pieces, y=train_labels)
    np.savez(path_test_outfile, x=test_pieces, y=test_labels)
