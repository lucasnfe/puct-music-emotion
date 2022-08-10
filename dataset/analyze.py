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
import argparse
import numpy as np

import pretty_midi
from utils import traverse_dir

def proc_one(path_infile):
    print('----')
    print(' >', path_infile)

    midi_obj = pretty_midi.PrettyMIDI(path_infile)
    return midi_obj.get_tempo_changes()[1], midi_obj.get_end_time()

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='augment.py')
    parser.add_argument('--path_indir', type=str, required=True)
    args = parser.parse_args()

    # list files
    midifiles = traverse_dir(
        args.path_indir,
        is_pure=True,
        is_sort=True)
    n_files = len(midifiles)
    print('num files:', n_files)

    # collect
    tempo_data = []
    duration_data = []

    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        print('{}/{}'.format(fidx, n_files))

        # paths
        path_infile = os.path.join(args.path_indir, path_midi)
        tempo_changes, duration = proc_one(path_infile)

        tempo_data += [t for t in tempo_changes]
        duration_data += [duration]

    print(np.mean(tempo_data), np.std(tempo_data))
    print(np.mean(duration_data), np.std(duration_data))
