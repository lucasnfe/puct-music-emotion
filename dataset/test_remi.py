import os
import json
import pickle
import argparse
import numpy as np
import multiprocessing as mp

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

from utils import *

# ================================ #
BEAT_RESOL = 1024
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {'piano': 0, 'melody': 1}

def write_midi(words, path_midi, word2event):
    notes_all = []

    events = [word2event[words[i]] for i in range(len(words))]

    bar_cnt = 0
    cur_beat = 0

    midi_obj = miditoolkit.midi.parser.MidiFile(ticks_per_beat=BEAT_RESOL)
    cur_pos = 0

    for i in range(len(events)-3):
        cur_event = events[i]
        # print(cur_event)
        name = cur_event.split('_')[0]
        attr = cur_event.split('_')
        if name == 'Bar':
            bar_cnt += 1
        elif name == 'Beat':
            cur_beat = int(attr[1])
            cur_pos = bar_cnt * BAR_RESOL + cur_beat * TICK_RESOL
        elif name == 'Chord':
            chord_text = attr[1] + '_' + attr[2]
            midi_obj.markers.append(Marker(text=chord_text, time=cur_pos))
        elif name == 'Tempo':
            midi_obj.tempo_changes.append(
                TempoChange(tempo=int(attr[1]), time=cur_pos))
        else:
            if 'Note_Pitch' in events[i] and \
            'Note_Velocity' in events[i+1] and \
            'Note_Duration' in events[i+2]:

                pitch = int(events[i].split('_')[-1])
                duration = int(events[i+2].split('_')[-1])

                if int(duration) == 0:
                    duration = 60

                end = cur_pos + duration
                velocity = int(events[i+1].split('_')[-1])
                notes_all.append(
                    Note(pitch=pitch, start=cur_pos, end=end, velocity=velocity))

    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = notes_all
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_midi)

def test(path_infile, path_outfile, word2event):
    # mkdir
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)

    # Load words
    words = np.load(path_infile)

    write_midi(words, '{}.mid'.format(path_outfile.split('.')[0]), word2event)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='compile_remi.py')
    parser.add_argument('--path_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    parser.add_argument('--path_dict', type=str, required=True)
    args = parser.parse_args()

    # paths
    os.makedirs(args.path_outdir, exist_ok=True)

    # load dictionary
    event2word, word2event = pickle.load(open(args.path_dict, 'rb'))

    # load all words
    wordfiles = traverse_dir(args.path_indir, is_pure=True, extension=('npy'))

    n_files = len(wordfiles)
    print('num fiels:', n_files)

    # collect
    data = []
    for fidx in range(n_files):
        path_midi = wordfiles[fidx]
        print('{}/{}'.format(fidx, n_files))

        # paths
        path_infile = os.path.join(args.path_indir, path_midi)
        path_outfile = os.path.join(args.path_outdir, path_midi)

        # append
        data.append([path_infile, path_outfile, word2event])

    # run, multi-thread
    pool = mp.Pool()
    pool.starmap(test, data)
