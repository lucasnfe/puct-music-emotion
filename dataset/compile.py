import os
import argparse
import numpy as np

from encoder import *
from utils import traverse_dir

def load_events_idx(path_infile, bar_token):
    events_idx = []
    with open(path_infile) as f:
        bars = []
        
        b = []
        for idx in f.read().split():
            b.append(int(idx))
            if int(idx) == bar_token:
                bars.append(b)
                b = []
   
        # Add end event to last bar
        bars[-1] += b

    return bars

def compile(path_indir, max_len, trim=False):
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
    bar_token = Event(event_type='control', value=1).to_int()

    pieces = []
    labels = []
    for fidx in range(n_files):
        path_txt = txtfiles[fidx]
        print('{}/{}'.format(fidx, path_txt))

        # Load events
        path_infile = os.path.join(path_indir, path_txt)
        bars = load_events_idx(path_infile, bar_token)

        # Load emotion
        emotion = process_emotion(path_txt)

        # Split the piece into sequences of len seq_len
        if trim:
            i = 0

            #while i < len(bars):
            piece = []
            while i < len(bars) and len(piece) + len(bars[i]) <= max_len:
                piece += bars[i]
                i += 1

            piece += [pad_token] * (max_len - len(piece))
            print(piece)

            pieces.append(piece)
            labels.append([emotion])
        else:
            pass
                

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
    parser.add_argument('--trim', action='store_true')
    parser.set_defaults(trim=False)
    args = parser.parse_args()

    os.makedirs(args.path_outdir, exist_ok=True)

    # Load datasets
    train_pieces, train_labels = compile(args.path_train_indir, args.max_len, args.trim)
    test_pieces, test_labels = compile(args.path_test_indir, args.max_len, args.trim)

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
