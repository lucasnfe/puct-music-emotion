#
# REMI Encoder
#
# Author: Lucas N. Ferreira - lucasnfe@gmail.com
#
# Base code: https://github.com/YatingMusic/compound-word-transformer
#

import os
import math
import argparse
import collections
import multiprocessing as mp
import numpy as np

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

from chorder import Dechorder, Chord

# ================================================== #
#  Configuration                                     #
# ================================================== #
INSTRUMENT_MAP = {
    'piano': 0
}

EMOTION_MAP = {
    'e0' : 0,
    'e1' : 1,
    'e2' : 2,
    'e3' : 3,
    'e4' : 4
}

DEGREE2PITCH = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
}

BEAT_RESOL = 1024
BAR_RESOL  = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

DEFAULT_NOTE_RANGE = 8
DEFAULT_NOTE_DOTS  = 4
DEFAULT_VELOCITY_BINS = np.linspace(40, 128, 64+1, dtype=np.int)
DEFAULT_TEMPO_BINS    = np.linspace(32, 224, 64+1, dtype=np.int)
DEFAULT_SHIFT_BINS    = np.linspace(-60, 60, 60+1, dtype=np.int)
DEFAULT_PITCH_BINS    = np.arange(0, 128, dtype=np.int)

N_BEAT     =  BAR_RESOL//TICK_RESOL
N_TEMPO    =  len(DEFAULT_TEMPO_BINS)
N_CHORD    =  len(DEGREE2PITCH) * len(Chord.standard_qualities)
N_VELOCITY =  len(DEFAULT_VELOCITY_BINS)
N_DURATION =  DEFAULT_NOTE_DOTS * (DEFAULT_NOTE_RANGE - 1)
N_PITCH    =  len(DEFAULT_PITCH_BINS)
N_EMOTION  =  len(EMOTION_MAP)
N_CONTROL  =  4 # START, STOP, BAR, PAD

VOCAB_SIZE = N_BEAT + N_TEMPO + N_CHORD + N_VELOCITY + N_DURATION + N_PITCH + N_EMOTION + N_CONTROL

start_idx = {
    'beat'    : 0,
    'tempo'   : N_BEAT,
    'chord'   : N_BEAT + N_TEMPO,
    'velocity': N_BEAT + N_TEMPO + N_CHORD,
    'duration': N_BEAT + N_TEMPO + N_CHORD + N_VELOCITY,
    'pitch'   : N_BEAT + N_TEMPO + N_CHORD + N_VELOCITY + N_DURATION,
    'emotion' : N_BEAT + N_TEMPO + N_CHORD + N_VELOCITY + N_DURATION + N_PITCH,
    'control' : N_BEAT + N_TEMPO + N_CHORD + N_VELOCITY + N_DURATION + N_PITCH + N_EMOTION
}

class Event:
    def __init__(self, event_type, value=0):
        assert event_type in start_idx, "Event " + event_type + ' not in start index.'
        self.type = event_type
        self.value = value

    def __repr__(self):
        return '<Event type: {}, value: {}>'.format(self.type, self.value)

    def to_int(self):
        return start_idx[self.type] + self.value

    @staticmethod
    def from_int(int_value):
        info = Event._type_check(int_value)
        return Event(info['type'], info['value'])

    @staticmethod
    def _type_check(int_value):
        valid_value = int_value

        if int_value in range(start_idx['beat'], start_idx['tempo']):
            return {'type': 'beat', 'value': valid_value}

        elif int_value in range(start_idx['tempo'], start_idx['chord']):
            valid_value -= start_idx['tempo']
            return {'type': 'tempo', 'value': valid_value}

        elif int_value in range(start_idx['chord'], start_idx['velocity']):
            valid_value -= start_idx['chord']
            return {'type': 'chord', 'value': valid_value}

        elif int_value in range(start_idx['velocity'], start_idx['duration']):
            valid_value -= start_idx['velocity']
            return {'type': 'velocity', 'value': valid_value}

        elif int_value in range(start_idx['duration'], start_idx['pitch']):
            valid_value -= start_idx['duration']
            return {'type': 'duration', 'value': valid_value}

        elif int_value in range(start_idx['pitch'], start_idx['emotion']):
            valid_value -= start_idx['pitch']
            return {'type': 'pitch', 'value': valid_value}

        elif int_value in range(start_idx['emotion'], start_idx['control']):
            valid_value -= start_idx['emotion']
            return {'type': 'emotion', 'value': valid_value}

        valid_value -= start_idx['control']
        return {'type': 'control', 'value': valid_value}


def _get_duration_values(note_range=DEFAULT_NOTE_RANGE, dots=DEFAULT_NOTE_DOTS, tempo=120):
    note_types = []

    # Generate all possible notes
    note_length = int(note_range * BEAT_RESOL)

    while note_length >= BEAT_RESOL//note_range:
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

def decode_midi(idx_array, path_outfile=None):
    # Create mid object
    midi_obj = miditoolkit.midi.parser.MidiFile(ticks_per_beat=BEAT_RESOL)
    events = [Event.from_int(idx) for idx in idx_array]

    bar_cnt = 0
    cur_pos = 0

    tempo = 120
    velocity = 0
    duration = 0

    notes = []
    for ev in events:
        if ev.type == "beat":
            cur_pos = bar_cnt * BAR_RESOL + ev.value * TICK_RESOL

        elif ev.type == "tempo":
            tempo = DEFAULT_TEMPO_BINS[ev.value]
            midi_obj.tempo_changes.append(TempoChange(tempo=tempo, time=cur_pos))

        elif ev.type == "velocity":
            velocity = DEFAULT_VELOCITY_BINS[ev.value]

        elif ev.type == "duration":
            note_values = _get_duration_values(tempo=tempo)
            duration = note_values[ev.value]

        elif ev.type == "pitch":
            note = Note(pitch=ev.value, start=cur_pos, end=cur_pos + duration, velocity=velocity)
            notes.append(note)

        elif ev.type == "control":
            if ev.value == 1:
                bar_cnt += 1
            elif ev.value == 2:
                break
            elif ev.value == 3:
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

def process_emotion(path_infile):
    path_basename = os.path.basename(path_infile)
    return EMOTION_MAP[path_basename.split('_')[0]]

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='encoder.py')
    parser.add_argument('--piece', type=str, required=True)
    parser.add_argument('--save_to', type=str, required=True)
    args = parser.parse_args()

    piece = [int(c) for c in args.piece.split(',')]

    decode_midi(piece, args.save_to)
    print(piece)
