import os
import argparse
import numpy as np
import multiprocessing as mp

from encoder import decode_midi
from utils import traverse_dir

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='compile.py')
    parser.add_argument('--path_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.path_outdir, exist_ok=True)

    # list files
    txtfiles = traverse_dir(
        args.path_indir,
        is_pure=True,
        is_sort=True,
        extension=('txt'))
    n_files = len(txtfiles)
    print('num files:', n_files)

    # collect
    data = []
    for fidx in range(n_files):
        path_txt = txtfiles[fidx]
        print('{}/{}'.format(fidx, n_files))

        # paths
        path_infile = os.path.join(args.path_indir, path_txt)
        path_outfile = os.path.join(args.path_outdir, path_txt)

        out_filename, _ = os.path.splitext(path_outfile)
        path_outfile = '{}.mid'.format(out_filename)

        # read data
        with open(path_infile) as f:
            events = [int(token) for token in f.read().split()]

        # append
        data.append([events, path_outfile])

    # run, multi-thread
    pool = mp.Pool()
    pool.starmap(decode_midi, data)

    # for d in data:
    #     print(d[0], d[1])
    #     decode_midi(d[0], d[1])
