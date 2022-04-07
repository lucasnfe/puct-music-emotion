import os
import pickle
import argparse
import numpy as np

from utils import *

def build_dict(path_indir, eventfiles, path_dict):
    # --- generate dictionary --- #
    print(' [*] generating dictionary')
    all_events = []
    for file in eventfiles:
        for event in pickle.load(open(os.path.join(path_indir, file), 'rb')):
            all_events.append('{}_{}'.format(event['name'], event['value']))

    # build
    unique_events = sorted(set(all_events), key=lambda x: (not isinstance(x, int), x))
    event2word = {key: i for i, key in enumerate(unique_events)}
    word2event = {i: key for i, key in enumerate(unique_events)}
    print(' > num classes:', len(word2event))

    # save
    pickle.dump((event2word, word2event), open(path_dict, 'wb'))

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='events_remi2words.py')
    parser.add_argument('--path_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    parser.add_argument('--path_dict', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.path_outdir, exist_ok=True)

    # list files
    eventfiles = traverse_dir(
        args.path_indir,
        is_pure=True,
        is_sort=True,
        extension=('pkl'))
    n_files = len(eventfiles)
    print('num fiels:', n_files)

    # --- build dictionary --- #
    # all files
    if not os.path.exists(args.path_dict):
        build_dict(args.path_indir, eventfiles, args.path_dict)

    # --- converts to word --- #
    event2word, word2event = pickle.load(open(args.path_dict, 'rb'))
    for fidx, file in enumerate(eventfiles):
        print('{}/{}'.format(fidx, n_files))

        # events to words
        path_infile = os.path.join(args.path_indir, file)
        events = pickle.load(open(path_infile, 'rb'))

        words = []
        for event in events:
            word = event2word['{}_{}'.format(event['name'], event['value'])]
            words.append(word)

        # save
        path_outfile = os.path.join(args.path_outdir, file + '.npy')
        fn = os.path.basename(path_outfile)
        os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
        np.save(path_outfile, words)
