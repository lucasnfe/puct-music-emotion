import os
import argparse
import numpy as np

from encoder import *
from utils import traverse_dir

def load_events_per_bar(path_infile, bar_token):
    bars = []
    
    with open(path_infile) as f:    
        b = []
        for idx in f.read().split():
            b.append(int(idx))
            if int(idx) == bar_token:
                bars.append(b)
                b = []
   
        # Add end event to last bar
        bars[-1] += b

    return bars

def load_events(path_infile):
    events_idx = []
    with open(path_infile) as f:
        events_idx = [int(idx) for idx in f.read().split()]
    return events_idx

def compile(path_indir, max_len, task='language_modeling'):
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

        path_infile = os.path.join(path_indir, path_txt)

        # Load emotion
        emotion = process_emotion(path_txt) - 1
        
        # Load origin (real vs fake)
        origin = process_origin(path_txt)

        # Split the piece into sequences of len seq_len
        if task == 'emotion_classification' or task == 'discriminator':
            i = 0
            
            # Load events
            bars = load_events_per_bar(path_infile, bar_token)

            piece = []
            while i < len(bars) and len(piece) + len(bars[i]) <= max_len:
                piece += bars[i]
                
                i += 1
                pieces.append(piece + [pad_token] * (max_len - len(piece)))
                
                if task == 'emotion_classification':
                    labels.append([emotion])
                elif task == 'discriminator':
                    labels.append([origin])
                else:
                    raise ValueError('Invalid task.')
        
        elif task == 'language_modeling':
            # Load events
            events = load_events(path_infile)
            
            # Split the piece into sequences of len seq_len
            for i in range(0, len(events), max_len):
                sequence = events[i:i+max_len]
                if len(sequence) < max_len:
                    # Pad sequence
                    sequence += [pad_token] * (max_len - len(sequence))

                pieces.append(sequence)
        else:
            raise ValueError('Invalid task.')

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
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.path_outdir, exist_ok=True)

    # Load datasets
    train_pieces, train_labels = compile(args.path_train_indir, args.max_len, args.task)
    test_pieces, test_labels = compile(args.path_test_indir, args.max_len, args.task)

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
