import os
import json
import pickle
import argparse
import numpy as np

from utils import *

TEST_AMOUNT = 50
WINDOW_SIZE = 512
GROUP_SIZE = 4
MAX_LEN = WINDOW_SIZE * GROUP_SIZE
COMPILE_TARGET = 'linear' # 'linear', 'XL'
print('[config] MAX_LEN:', MAX_LEN)

def compile(path_indir, path_outdir, path_dict, prefix_out):
    # load dictionary
    event2word, word2event = pickle.load(open(path_dict, 'rb'))
    eos_id = event2word['EOS_None']
    print(' > eos_id:', eos_id)

    # load all words
    wordfiles = traverse_dir(path_indir, extension=('npy'))

    # load dictionary
    event2word, word2event = pickle.load(open(path_dict, 'rb'))
    eos_id = event2word['EOS_None']
    print(' > eos_id:', eos_id)

    # load all words
    wordfiles = traverse_dir(path_indir, extension=('npy'))
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
        print('--[{}/{}]-----'.format(fidx, n_files))
        file = wordfiles[fidx]
        words = np.load(file)
        num_words = len(words)

        if num_words >= MAX_LEN - 2: # 2 for room
            print(' [!] too long:', num_words)
            continue

        # arrange IO
        x = words[:-1]
        y = words[1:]
        seq_len = len(x)
        print(' > seq_len:', seq_len)

        # pad with eos
        x = np.concatenate([x, np.ones(MAX_LEN-seq_len) * eos_id])
        y = np.concatenate([y, np.ones(MAX_LEN-seq_len) * eos_id])
        mask = np.concatenate(
            [np.ones(seq_len), np.zeros(MAX_LEN-seq_len)])

        # collect
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
        x_final = np.array(x_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
        y_final = np.array(y_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
        mask_final = np.array(mask_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
    elif COMPILE_TARGET == 'linear':
        x_final = np.array(x_list)
        y_final = np.array(y_list)
        mask_final = np.array(mask_list)
    else:
        raise ValueError('Unknown target:', COMPILE_TARGET)

    num_samples = len(seq_len_list)
    print(' >   count:', )
    print(' > x_final:', x_final.shape)
    print(' > y_final:', y_final.shape)
    print(' > mask_final:', mask_final.shape)

    # save
    path_train = os.path.join(path_outdir,
        '{}_data_remi_{}.npz'.format(prefix_out, COMPILE_TARGET))

    np.savez(
        path_train,
        x=x_final,
        y=y_final,
        mask=mask_final,
        seq_len=np.array(seq_len_list),
        num_groups=np.array(num_groups_list)
    )

    print('---')
    print(' > train x:', x_final.shape)
    print(' >  test x:', x_final.shape)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='compile_remi.py')
    parser.add_argument('--path_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    parser.add_argument('--path_dict', type=str, required=True)
    args = parser.parse_args()

    train_path_indir = os.path.join(args.path_indir, 'train')
    test_path_indir = os.path.join(args.path_indir, 'test')

    compile(train_path_indir, args.path_outdir, args.path_dict, "train")
    compile(test_path_indir, args.path_outdir, args.path_dict, "test")
