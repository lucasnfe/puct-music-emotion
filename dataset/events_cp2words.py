
'''
This code is modify from:
https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/representations/uncond/cp/events2words.py
'''

import os
import pickle
import argparse
import numpy as np
import collections

from utils import *

def build_dict(path_indir, eventfiles, path_dict):
    class_keys = pickle.load(
        open(os.path.join(path_indir, eventfiles[0]), 'rb'))[0].keys()
    print('class keys:', class_keys)

    # define dictionary
    event2word = {}
    word2event = {}

    corpus_kv = collections.defaultdict(list)
    for file in eventfiles:
        for event in pickle.load(open(
            os.path.join(path_indir, file), 'rb')):
            for key in class_keys:
                corpus_kv[key].append(event[key])

    for ckey in class_keys:
        class_unique_vals = sorted(
            set(corpus_kv[ckey]), key=lambda x: (not isinstance(x, int), x))
        event2word[ckey] = {key: i for i, key in enumerate(class_unique_vals)}
        word2event[ckey] = {i: key for i, key in enumerate(class_unique_vals)}

    # print
    print('[class size]')
    for key in class_keys:
        print(' > {:10s}: {}'.format(key, len(event2word[key])))

    # save
    pickle.dump((event2word, word2event), open(path_dict, 'wb'))

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='events2words.py')
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

    class_keys = pickle.load(
        open(os.path.join(args.path_indir, eventfiles[0]), 'rb'))[0].keys()
    print('class keys:', class_keys)

    # --- build dictionary --- #
    # all files
    if not os.path.exists(args.path_dict):
        build_dict(args.path_indir, eventfiles, args.path_dict)

    # --- compile each --- #
    # reload
    event2word, word2event = pickle.load(open(args.path_dict, 'rb'))
    for fidx in range(len(eventfiles)):
        file = eventfiles[fidx]
        events_list = pickle.load(open(
            os.path.join(args.path_indir, file), 'rb'))

        fn = os.path.basename(file)
        path_outfile = os.path.join(args.path_outdir, fn)

        print('({}/{})'.format(fidx, len(eventfiles)))
        print(' > from:', file)
        print(' >   to:', path_outfile)

        words = []
        for eidx, e in enumerate(events_list):
            words_tmp = [event2word[k][e[k]] for k in class_keys]
            words.append(words_tmp)

        # save
        path_outfile = os.path.join(args.path_outdir, '{}.npy'.format(file))
        fn = os.path.basename(path_outfile)
        os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
        np.save(path_outfile, words)
