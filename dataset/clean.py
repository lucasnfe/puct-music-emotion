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
import pretty_midi
import numpy as np
import multiprocessing as mp

from utils import traverse_dir

INSTR_NAME_MAP = {'piano': 0}

def clean(path_infile, path_outfile):
    print('----')
    print(' >', path_infile)
    print(' >', path_outfile)

    mid = pretty_midi.PrettyMIDI(midi_file=path_infile)

    for time_signature in mid.time_signature_changes:
        if time_signature.numerator   != 4 or \
           time_signature.denominator != 4:
            print("Midi has a non common time signature", time_signature)
            return

    first_piano_ix = None
    instruments_to_remove = []

    for i in range(len(mid.instruments)):
        if mid.instruments[i].is_drum:
            print("Midi has a percusion instrument.", mid.instruments[i])
            instruments_to_remove.append(mid.instruments[i])
            continue

        if pretty_midi.program_to_instrument_class(mid.instruments[i].program) != "Piano":
            print("Midi has a non-piano instrument.", mid.instruments[i])
            instruments_to_remove.append(mid.instruments[i])
            continue

        if first_piano_ix == None:
            first_piano_ix = i
        else:
            mid.instruments[first_piano_ix].notes += mid.instruments[i].notes
            mid.instruments[first_piano_ix].pitch_bends += mid.instruments[i].pitch_bends
            mid.instruments[first_piano_ix].control_changes += mid.instruments[i].control_changes

            instruments_to_remove.append(mid.instruments[i])

        mid.instruments[i].program = 0
        mid.instruments[i].name = 'piano'

    # remove tracks
    print("mid.instruments before", len(mid.instruments), instruments_to_remove)
    for instrument in instruments_to_remove:
        mid.instruments.remove(instrument)
    print("mid.instruments after", len(mid.instruments))

    # === save === #
    # mkdir
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)

    # save
    mid.write(path_outfile)

if __name__ == '__main__':
    # paths
    path_indir = './midi_raw'
    path_outdir = './midi_clean'

    os.makedirs(path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(
        path_indir,
        is_pure=True,
        is_sort=True)
    n_files = len(midifiles)
    print('num fiels:', n_files)

    # collect
    data = []
    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        print('{}/{}'.format(fidx, n_files))

        # paths
        path_infile = os.path.join(path_indir, path_midi)
        path_outfile = os.path.join(path_outdir, path_midi)

        # append
        data.append([path_infile, path_outfile])

    # run, multi-thread
    pool = mp.Pool()
    pool.starmap(clean, data)
