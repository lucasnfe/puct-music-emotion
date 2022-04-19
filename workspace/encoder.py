#
# Encoder based on PerformanceRNN & Music Transformer to process
# MIDI data with neural networks.
#
# Author: Lucas N. Ferreira - lucasnfe@gmail.com
#
# PerformanceRNN: https://magenta.tensorflow.org/performance-rnn
# Music Transformer: https://magenta.tensorflow.org/music-transformer
# Base code: https://github.com/jason9693/midi-neural-processor
#

import os
import math
import argparse
import collections
import multiprocessing as mp
import numpy as np

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

from chorder import Dechorder

# ================================================== #
#  Configuration                                     #
# ================================================== #
MIN_BPM = 40
MIN_VELOCITY = 40

INSTR_NAME_MAP = {'piano': 0}

DEFAULT_VELOCITY_BINS = np.linspace(0,  128, 64+1, dtype=np.int)
DEFAULT_TEMPO_BINS    = np.linspace(32, 224, 64+1, dtype=np.int)
DEFAULT_SHIFT_BINS    = np.linspace(-60, 60, 60+1, dtype=np.int)
DEFAULT_PITCH_BINS    = np.arange(0, 128, dtype=np.int)

def _get_duration_values(beat_resol, note_range=8, dots=4, tempo=120):
    note_types = []

    # Generate all possible notes
    note_length = int(note_range * beat_resol)

    while note_length >= beat_resol//note_range:
        # Append current note length (e.g. whole, half, quarter...)
        note_types.append(note_length)

        # Generate dot divisions
        dot_length = note_length//2
        dotted_note_length = note_length + dot_length
        for i in range(1, dots):
            # Append current dotted note
            note_types.append(dotted_note_length)

            dot_length = dot_length//2
            dotted_note_length = dotted_note_length + dot_length

        note_length = note_length//2

    return note_types

def decode_midi(idx_array, vocab, path_outfile=None, ticks_per_beat=1024):
    # Create mid object
    midi_obj = miditoolkit.midi.parser.MidiFile(ticks_per_beat=ticks_per_beat)

    beat_resol = ticks_per_beat
    bar_resol  = beat_resol * 4
    tick_resol = beat_resol // 4

    # load vocabulary
    idx2event = {i:event for event,i in vocab.items()}

    bar_cnt = 0
    cur_pos = 0

    notes = []
    for idx in idx_array:
        event = idx2event[idx].split('_')
        ev_type = event[0]

        if ev_type == 'b':
            ev_value = int(event[1])
            cur_pos = bar_cnt * bar_resol + ev_value * tick_resol

        elif ev_type == 't':
            ev_value = int(event[1])
            tempo = DEFAULT_TEMPO_BINS[ev_value]
            midi_obj.tempo_changes.append(TempoChange(tempo=tempo, time=cur_pos))

        elif ev_type == 'v':
            ev_value = int(event[1])
            velocity = DEFAULT_VELOCITY_BINS[ev_value]

        elif ev_type == 'd':
            ev_value = int(event[1])
            note_values = _get_duration_values(beat_resol=beat_resol, tempo=tempo)
            duration = note_values[ev_value]

        elif ev_type == 'p':
            ev_value = int(event[1])
            note = Note(pitch=ev_value, start=cur_pos, end=cur_pos + duration, velocity=velocity)
            notes.append(note)

        elif ev_type == '|':
            bar_cnt += 1
        
        elif ev_type == 'e':
            break

    # add events to  midi object
    piano = Instrument(0, is_drum=False, name='piano')
    piano.notes = notes
    midi_obj.instruments = [piano]

    # save midi
    if path_outfile:
        fn = os.path.basename(path_outfile)
        os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
        midi_obj.dump(path_outfile)

    return midi_obj

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='encoder.py')
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

        out_filename, _ = os.path.splitext(path_outfile)
        path_outfile = '{}.txt'.format(out_filename)

        # append
        data.append([path_infile, path_outfile])

    # run, multi-thread
    pool = mp.Pool()
    pool.starmap(encode_midi, data)

    # for d in data:
    #     encode_midi(d[0], d[1])
