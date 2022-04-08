import os
import argparse
import numpy as np

from utils import traverse_dir

PAD_TOKEN = 389

def load_events(path_infile):
    events = []
    with open(path_infile) as f:
        events = np.array([int(token) for token in f.read().split()])
    return events

def compile(path_indir, max_len):
    # list files
    txtfiles = traverse_dir(
        path_indir,
        is_pure=True,
        is_sort=True,
        extension=("txt"))
    n_files = len(txtfiles)
    print('num files:', n_files)

    dataset = []
    for fidx in range(n_files):
        path_txt = txtfiles[fidx]
        print('{}/{}'.format(fidx, path_txt))

        # Load events
        path_infile = os.path.join(path_indir, path_txt)
        events = load_events(path_infile)

        # Compute number of sequences to create from events
        n_sequences = events.shape[0]//args.max_len
        n_leftovers = (n_sequences + 1) * args.max_len - events.shape[0]

        # Pad events before splitting sequences
        padded_events = np.pad(events, (0, n_leftovers), constant_values=(PAD_TOKEN))

        sequences = padded_events.reshape(n_sequences + 1, max_len)
        dataset.append(sequences)

    dataset = np.vstack(dataset)
    return dataset

def create_labels(dataset):
    x = dataset[:-1]
    y = dataset[1:]
    mask = (dataset != PAD_TOKEN).astype(int)

    return x, y, mask

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='compile.py')
    parser.add_argument('--path_train_indir', type=str, required=True)
    parser.add_argument('--path_test_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    parser.add_argument('--max_len', type=int, required=True)
    args = parser.parse_args()

    # Load datasets
    train_data = compile(args.path_train_indir, args.max_len)
    test_data = compile(args.path_test_indir, args.max_len)

    print('---')
    print(' > train x:', train_data.shape)
    print(' >  test x:', test_data.shape)

    x_train, y_train, mask_train = create_labels(train_data)
    x_test, y_test, mask_test = create_labels(test_data)

    # Save datasets
    path_train_outfile = os.path.join(args.path_outdir, 'train.npz')
    path_test_outfile = os.path.join(args.path_outdir, 'test.npz')

    np.savez(path_train_outfile, x=x_train, y=y_train, mask=mask_train)
    np.savez(path_test_outfile, x=x_test, y=y_test, mask=mask_test)
