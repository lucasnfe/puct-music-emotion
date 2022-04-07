'''
This code is from
https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/representations/uncond/cp/compile.py
'''

import os
import json
import pickle
import argparse
import numpy as np
import ipdb

TEST_AMOUNT = 50
WINDOW_SIZE = 512
GROUP_SIZE = 4    #7
MAX_LEN = WINDOW_SIZE * GROUP_SIZE
COMPILE_TARGET = 'linear' # 'linear', 'XL'
print('[config] MAX_LEN:', MAX_LEN)

from utils import *

def compile(path_indir, path_outdir, prefix_out):
    # load all words
    wordfiles = traverse_dir(
            path_indir,
            extension=('npy'))
    n_files = len(wordfiles)

    # init
    x_list = []
    y_list = []
    mask_list = []
    seq_len_list = []
    num_groups_list = []
    name_list = []

    # process
    for fidx in range(n_files):
        print('--[{}/{}]-----'.format(fidx+1, n_files))

        file = wordfiles[fidx]
        print(file)

        try:
            words = np.load(file)
        except:
            print(fidx)

            import ipdb
            ipdb.set_trace()

        num_words = len(words)
        eos_arr = words[-1][None, ...]
        if num_words >= MAX_LEN - 2: # 2 for room
            words = words[:MAX_LEN-2]

        # arrange IO
        x = words[:-1].copy()  #without EOS
        y = words[1:].copy()
        seq_len = len(x)
        print(' > seq_len:', seq_len)

        # pad with eos
        pad = np.tile(eos_arr, (MAX_LEN-seq_len, 1))

        x = np.concatenate([x, pad], axis=0)
        y = np.concatenate([y, pad], axis=0)
        mask = np.concatenate(
            [np.ones(seq_len), np.zeros(MAX_LEN-seq_len)])

        # collect
        if x.shape != (MAX_LEN, 8):
            print(x.shape)
            exit()

        x_list.append(x)
        y_list.append(y)
        mask_list.append(mask)
        seq_len_list.append(seq_len)
        num_groups_list.append(int(np.ceil(seq_len/WINDOW_SIZE)))
        name_list.append(file)

    # sort by length (descending)
    zipped = zip(seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list)
    seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list = zip(
                                    *sorted(zipped, key=lambda x: -x[0]))

    print('\n\n[Finished]')
    print(' compile target:', COMPILE_TARGET)
    if COMPILE_TARGET == 'XL':
        # reshape
        x_final = np.array(x_list).reshape(len(x_list), GROUP_SIZE, WINDOW_SIZE, -1)
        y_final = np.array(y_list).reshape(len(x_list), GROUP_SIZE, WINDOW_SIZE, -1)
        mask_final = np.array(mask_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
    elif COMPILE_TARGET == 'linear':

        x_final = np.array(x_list)
        y_final = np.array(y_list)
        mask_final = np.array(mask_list)
    else:
        raise ValueError('Unknown target:', COMPILE_TARGET)

    # check
    num_samples = len(seq_len_list)
    print(' >   count:', )
    print(' > x_final:', x_final.shape)
    print(' > y_final:', y_final.shape)
    print(' > mask_final:', mask_final.shape)

    train_idx = []

    # training filename map
    train_fn2idx_map = {
        'fn2idx': dict(),
        'idx2fn': dict(),
    }

    name_list = [x.split('/')[-1].split('.')[0] for x in name_list]

    # run split
    train_cnt = 0
    for nidx, n in enumerate(name_list):
        train_idx.append(nidx)
        train_fn2idx_map['fn2idx'][n] = train_cnt
        train_fn2idx_map['idx2fn'][train_cnt] = n
        train_cnt += 1

    train_idx = np.array(train_idx)

    # save train map
    path_train_fn2idx_map = os.path.join(path_outdir,
        '{}_fn2idx_map.json'.format(prefix_out))

    with open(path_train_fn2idx_map, 'w') as f:
        json.dump(train_fn2idx_map, f)

    # save train
    path_train = os.path.join(path_outdir,
        '{}_data_cp_{}.npz'.format(prefix_out, COMPILE_TARGET))

    print('save to', path_train)

    np.savez(
        path_train,
        x=x_final[train_idx],
        y=y_final[train_idx],
        mask=mask_final[train_idx],
        seq_len=np.array(seq_len_list)[train_idx],
        num_groups=np.array(num_groups_list)[train_idx]
    )

    print('---')
    print(' > {} x:'.format(prefix_out), x_final[train_idx].shape)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='compile_cp.py')
    parser.add_argument('--path_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    args = parser.parse_args()

    train_path_indir = os.path.join(args.path_indir, 'train')
    test_path_indir = os.path.join(args.path_indir, 'test')

    compile(train_path_indir, args.path_outdir, 'train')
    compile(test_path_indir, args.path_outdir, 'test')
