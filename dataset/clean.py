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
import csv
import argparse
import pretty_midi
import numpy as np
import multiprocessing as mp

from utils import traverse_dir

EMOTION_MAP = {
    ( 0,  0) :'e0',
    ( 1,  1) :'e1',
    (-1,  1) :'e2',
    (-1, -1) :'e3',
    ( 1, -1) :'e4'
}

def clean(path_infile, path_outfile, emotion_annotation=None):
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
    # Parse arguments
    parser = argparse.ArgumentParser(description='clean.py')
    parser.add_argument('--path_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    parser.add_argument('--path_emotion', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(
        args.path_indir,
        is_pure=True,
        is_sort=True)
    n_files = len(midifiles)
    print('num files:', n_files)

    # Load emotion data
    emotion_annotation = None
    if args.path_emotion:
        emotion_annotation = {}
        for row in csv.DictReader(open(args.path_emotion, "r")):
            emotion_annotation[row['midi']] = (int(row['valence']), int(row['arousal']))

    # collect
    data = []
    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        print('{}/{}'.format(fidx, n_files))

        # Get piece emotion
        emotion = (0, 0)
        if path_midi in emotion_annotation:
            emotion = emotion_annotation[path_midi]

        path_midi_basename = os.path.basename(path_midi)
        path_midi_emotion = path_midi.replace(path_midi_basename, '{}_{}'.format(EMOTION_MAP[emotion], path_midi_basename))

        # paths
        path_infile = os.path.join(args.path_indir, path_midi)
        path_outfile = os.path.join(args.path_outdir, path_midi_emotion)

        # append
        data.append([path_infile, path_outfile, emotion_annotation])

    # run, multi-thread
    pool = mp.Pool()
    pool.starmap(clean, data)

    # for d in data:
    #     clean(d[0], d[1], d[2])
