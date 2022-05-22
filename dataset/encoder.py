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
from utils import traverse_dir

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

ORIGIN_MAP = {
    'fake' : 0,
    'real' : 1
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

def _load_notes(midi_obj, note_sorting = 1):
    # load notes
    instr_notes = collections.defaultdict(list)

    for instr in midi_obj.instruments:
        # skip
        if instr.name not in INSTRUMENT_MAP.keys():
            continue

        # process
        instr_idx = INSTRUMENT_MAP[instr.name]
        for note in instr.notes:
            note.instr_idx = instr_idx
            instr_notes[instr_idx].append(note)
        if note_sorting == 0:
            instr_notes[instr_idx].sort(key=lambda x: (x.start, x.pitch))
        elif note_sorting == 1:
            instr_notes[instr_idx].sort(key=lambda x: (x.start, -x.pitch))
        else:
            raise ValueError(' [x] Unknown type of sorting.')

    return instr_notes

def _load_tempo_changes(midi_obj):
    tempi = midi_obj.tempo_changes
    tempi.sort(key=lambda x: x.time)
    return tempi

def _load_chords(midi_obj):
    chords = []
    for cidx, chord in enumerate(Dechorder.dechord(midi_obj)):
        if chord.root_pc is not None and chord.quality is not None:
            chord_text = '{}_{}'.format(DEGREE2PITCH[chord.root_pc], chord.quality)
            chords.append(Marker(time=int(cidx * BEAT_RESOL), text=chord_text))

    return chords

def _process_notes(notes, tempo_changes, offset):
    intsr_grid = dict()

    for key in notes.keys():
        note_grid = collections.defaultdict(list)
        for note in notes[key]:
            note.start = note.start - offset * BAR_RESOL
            note.end = note.end - offset * BAR_RESOL

            # quantize start
            quant_time = int(np.round(note.start / TICK_RESOL) * TICK_RESOL)

            # velocity
            note.velocity = np.argmin(abs(DEFAULT_VELOCITY_BINS - note.velocity))

            # shift of start
            note.shift = note.start - quant_time
            note.shift = np.argmin(abs(DEFAULT_SHIFT_BINS - note.shift))

            # duration
            note_tempo = _get_note_tempo(note, tempo_changes)
            note_duration = note.end - note.start
            note.duration = _get_closest_note_value(note_duration, tempo=note_tempo)

            # append
            note_grid[quant_time].append(note)

        # set to track
        intsr_grid[key] = note_grid.copy()

    return intsr_grid

def process_emotion(path_infile):
    path_basename = os.path.basename(path_infile)
    return EMOTION_MAP[path_basename.split('_')[0]]

def process_origin(path_infile):
    path_basename = os.path.basename(path_infile)
    return ORIGIN_MAP[path_basename.split('_')[1]]

def _process_tempo_changes(tempo_changes, offset):
    tempo_grid = collections.defaultdict(list)
    for tempo in tempo_changes:
        # quantize
        tempo.time = tempo.time - offset * BAR_RESOL
        tempo.time = 0 if tempo.time < 0 else tempo.time

        quant_time = int(np.round(tempo.time / TICK_RESOL) * TICK_RESOL)
        tempo.tempo = np.argmin(abs(DEFAULT_TEMPO_BINS-tempo.tempo))

        # append
        tempo_grid[quant_time].append(tempo)

    return tempo_grid

def _process_chords(chords, offset):
    chord_map = _get_chord_map()

    chord_grid = collections.defaultdict(list)
    for chord in chords:
        # quantize
        chord.time = chord.time - offset * BAR_RESOL
        chord.time = 0 if chord.time < 0 else chord.time
        chord.text = chord_map[chord.text]
        quant_time = int(np.round(chord.time / TICK_RESOL) * TICK_RESOL)

        # append
        chord_grid[quant_time].append(chord)

    return chord_grid

def _create_events(notes, tempo_changes, chords, last_bar, emotion=0):
    events = []

    # Start of piece event
    events.append(Event(event_type='control', value=0))
    events.append(Event(event_type='emotion', value=emotion))

    for bar_step in range(0, last_bar * BAR_RESOL, BAR_RESOL):
        # --- piano track --- #
        for t in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            t_tempos = tempo_changes[t]
            t_notes  = notes[0][t]
            t_chords = chords[t]

            # Beat
            beat_value = (t - bar_step)//TICK_RESOL
            events.append(Event(event_type='beat', value=beat_value))

            # Tempo
            if len(t_tempos):
                events.append(Event(event_type='tempo', value=t_tempos[-1].tempo))

            # Chord
            if len(t_chords):
                events.append(Event(event_type='chord', value=t_chords[-1].text))

            # Notes
            for note in t_notes:
                events.append(Event(event_type='velocity', value=note.velocity))
                events.append(Event(event_type='duration', value=note.duration))
                events.append(Event(event_type='pitch', value=note.pitch))

        # create bar event
        events.append(Event(event_type='control', value=1))

    # End of piece event
    events.append(Event(event_type='control', value=2))

    return [ev.to_int() for ev in events]

def _get_note_tempo(note, tempo_changes):
    i = 0
    while i < len(tempo_changes) and note.start >= tempo_changes[i].time:
        i += 1
    return tempo_changes[i - 1].tempo

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

def _get_chord_map():
    chord_map = {}
    chord_idx = 0

    for degree, pitch in DEGREE2PITCH.items():
        for quality in Chord.standard_qualities:
            chord = '{}_{}'.format(pitch, quality)
            chord_map[chord] = chord_idx
            chord_idx += 1

    return chord_map

def _get_closest_note_value(delta_time, tempo):
    note_types = _get_duration_values(tempo=tempo)

    min_dist = math.inf
    min_type = None
    for i, type in enumerate(note_types):
        dist = abs(delta_time - type)
        if dist < min_dist:
            min_dist = dist
            min_type = i

    return min_type

def encode_midi(path_infile, path_outfile, include_emotion=False, note_sorting=1):
    # --- load --- #
    midi_obj = miditoolkit.midi.parser.MidiFile(path_infile)

    assert midi_obj.ticks_per_beat == BEAT_RESOL

    # notes and tempo changes
    notes = _load_notes(midi_obj, note_sorting=note_sorting)
    tempo_changes = _load_tempo_changes(midi_obj)
    chords = _load_chords(midi_obj)

    # --- process items to grid --- #
    # compute empty bar offset at head
    first_note_time = min([notes[k][0].start for k in notes.keys()])
    last_note_time = max([notes[k][-1].start for k in notes.keys()])

    # compute quantized time of the first note
    quant_time_first = int(np.round(first_note_time / TICK_RESOL) * TICK_RESOL)

    # compute quantized offset and last bar time
    offset = quant_time_first // BAR_RESOL
    last_bar = int(np.ceil(last_note_time / BAR_RESOL)) - offset
    
    print(path_infile)
    print(' > offset:', offset)
    print(' > last_bar:', last_bar)

    # process quantized notes and tempo
    note_grid = _process_notes(notes, tempo_changes, offset)
    tempo_grid = _process_tempo_changes(tempo_changes, offset)
    chord_grid = _process_chords(chords, offset)
    
    emotion = 0
    if include_emotion:
        emotion = process_emotion(path_infile)

    # create events
    events = _create_events(note_grid, tempo_grid, chord_grid, last_bar, emotion)

    # save
    if path_outfile:
        fn = os.path.basename(path_outfile)
        os.makedirs(path_outfile[:-len(fn)], exist_ok=True)

        with open(path_outfile, 'w') as f:
            f.write(' '.join([str(e) for e in events]))

    return events

def decode_midi(idx_array, path_outfile=None):
    # Create mid object
    midi_obj = miditoolkit.midi.parser.MidiFile(ticks_per_beat=BEAT_RESOL)
    events = [Event.from_int(idx) for idx in idx_array]

    bar_cnt = 0
    cur_pos = 0

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

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='encoder.py')
    parser.add_argument('--path_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    parser.add_argument('--include_emotion', action='store_true')
    parser.set_defaults(include_emotion=False)
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
        data.append([path_infile, path_outfile, args.include_emotion])

    # run, multi-thread
    pool = mp.Pool()
    pool.starmap(encode_midi, data)

    # for d in data:
    #     encode_midi(d[0], d[1])
