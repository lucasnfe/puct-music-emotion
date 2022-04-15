#
# Clean MIDI data to:
# 1) Remove all pieces that are not in common time (i.e., 4/4)
# 2) Remove all non-piano tracks
# 3) Combine multiple piano tracks into a single one
#
# Lucas N. Ferreira
# lucasnfe@gmail.com
#
#

import os
import copy
import argparse
import numpy as np
import multiprocessing as mp

import miditoolkit
from miditoolkit.midi.containers import Instrument, TempoChange, Note

from utils import traverse_dir

# Range of transposition intervals (in semi-tones)
transpose_intervals = range(-4, 5)

# Range of strech factors
stretch_factors = [0.8, 0.9, 1.0]

def transpose(midi_obj, interval):
    # Transpose all pitched notes
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += interval

def strech(midi_obj, stretch_factor):
    for tempo_change in midi_obj.tempo_changes:
        tempo_change.tempo = round(tempo_change.tempo * stretch_factor)


def proc_one(path_infile, path_outfile):
    print('----')
    print(' >', path_infile)

    midi_obj = miditoolkit.midi.parser.MidiFile(path_infile)

    # Transpose midi
    for interval in transpose_intervals:
        # Make a copy of the original midi
        transpose_mid = copy.deepcopy(midi_obj)

        # Transpose midi
        transpose(transpose_mid, interval)

        if interval < 0:
            interval_suffix = "_b_" + str(abs(interval))
        elif interval == 0:
            interval_suffix = "_original"
        elif interval > 0:
            interval_suffix = "_#_" + str(abs(interval))

        # Strech midi
        for factor in stretch_factors:
            transposed_streched_mid = copy.deepcopy(transpose_mid)

            # Strech midi
            strech(transposed_streched_mid, factor)

            if factor < 1.0:
                strech_suffix = "_slow_" + str(int(factor * 100))
            elif factor == 1.0:
                strech_suffix = "_original"
            elif factor > 1.0:
                strech_suffix = "_fast_" + str(int(factor * 100))

            name_outfile, _ = os.path.splitext(path_outfile)
            new_path_outfile = name_outfile + interval_suffix + strech_suffix + ".mid"

            # mkdir
            fn = os.path.basename(new_path_outfile)
            os.makedirs(new_path_outfile[:-len(fn)], exist_ok=True)

            # save
            print(' >', new_path_outfile)
            transposed_streched_mid.dump(new_path_outfile)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='augment.py')
    parser.add_argument('--path_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(
        args.path_indir,
        is_pure=True,
        is_sort=True)
    n_files = len(midifiles)
    print('num files:', n_files)

    # collect
    data = []
    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        print('{}/{}'.format(fidx, n_files))

        # paths
        path_infile = os.path.join(args.path_indir, path_midi)
        path_outfile = os.path.join(args.path_outdir, path_midi)

        # append
        data.append([path_infile, path_outfile])

    # run, multi-thread
    pool = mp.Pool()
    pool.starmap(proc_one, data)
