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
from utils import traverse_dir

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

def _load_notes(midi_obj, note_sorting = 1):
    # load notes
    instr_notes = collections.defaultdict(list)

    for instr in midi_obj.instruments:
        # skip
        if instr.name not in INSTR_NAME_MAP.keys():
            continue

        # process
        instr_idx = INSTR_NAME_MAP[instr.name]
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

def _load_chords(midi_obj, beat_resol):
    num2pitch = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }

    chords = []
    for cidx, chord in enumerate(Dechorder.dechord(midi_obj)):
        if chord.is_complete():
            chord_text = '{}_{}_{}'.format(num2pitch[chord.root_pc], chord.quality, num2pitch[chord.bass_pc])
        else:
            chord_text = 'N_N_N'

        chords.append(Marker(time=int(cidx * beat_resol), text=chord_text))

    return chords

def _process_notes(notes, tempo_changes, offset, tick_resol, beat_resol, bar_resol):
    intsr_grid = dict()

    for key in notes.keys():
        note_grid = collections.defaultdict(list)
        for note in notes[key]:
            note.start = note.start - offset * bar_resol
            note.end = note.end - offset * bar_resol

            # quantize start
            quant_time = int(np.round(note.start / tick_resol) * tick_resol)

            # velocity
            note.velocity = DEFAULT_VELOCITY_BINS[np.argmin(abs(DEFAULT_VELOCITY_BINS - note.velocity))]
            note.velocity = max(MIN_VELOCITY, note.velocity)

            # shift of start
            note.shift = note.start - quant_time
            note.shift = DEFAULT_SHIFT_BINS[np.argmin(abs(DEFAULT_SHIFT_BINS - note.shift))]

            # duration
            note_tempo = _get_note_tempo(note, tempo_changes)
            note_duration = note.end - note.start
            note.duration = _get_closest_note_value(note_duration, beat_resol, tempo=note_tempo)

            # append
            note_grid[quant_time].append(note)

        # set to track
        intsr_grid[key] = note_grid.copy()

    return intsr_grid

def _get_note_tempo(note, tempo_changes):
    i = 0
    while i < len(tempo_changes) and note.start >= tempo_changes[i].time:
        i += 1
    return tempo_changes[i - 1].tempo

def _process_tempo_changes(tempo_changes, offset, tick_resol, bar_resol):
    tempo_grid = collections.defaultdict(list)
    for tempo in tempo_changes:
        # quantize
        tempo.time = tempo.time - offset * bar_resol
        tempo.time = 0 if tempo.time < 0 else tempo.time

        quant_time = int(np.round(tempo.time / tick_resol) * tick_resol)
        tempo.tempo = DEFAULT_TEMPO_BINS[np.argmin(abs(DEFAULT_TEMPO_BINS-tempo.tempo))]

        # append
        tempo_grid[quant_time].append(tempo)

    return tempo_grid

def _process_chords(chords, offset, tick_resol, bar_resol):
    chord_grid = collections.defaultdict(list)
    for chord in chords:
        # quantize
        chord.time = chord.time - offset * bar_resol
        chord.time  = 0 if chord.time < 0 else chord.time
        quant_time = int(np.round(chord.time / tick_resol) * tick_resol)

        # append
        chord_grid[quant_time].append(chord)

    return chord_grid

def _create_events(notes, tempo_changes, chords, last_bar, tick_resol, bar_resol):
    events = []

    # End of piece event
    events.append("s")

    for bar_step in range(0, last_bar * bar_resol, bar_resol):
        # --- piano track --- #
        for t in range(bar_step, bar_step + bar_resol, tick_resol):
            t_tempos = tempo_changes[t]
            t_notes  = notes[0][t]
            t_chords = chords[t]

            # Beat
            beat_value = (t - bar_step)//tick_resol
            events.append('b_{}'.format(beat_value))

            # Tempo
            if len(t_tempos):
                tempo_value = int(np.where(DEFAULT_TEMPO_BINS == t_tempos[-1].tempo)[0])
                events.append('t_{}'.format(tempo_value))

            # Chord
            if len(t_chords):
                root, quality, _ = t_chords[-1].text.split('_')
                events.append('c_{}_{}'.format(root, quality))

            # Notes
            for note in t_notes:
                velocity_value = int(np.where(DEFAULT_VELOCITY_BINS == note.velocity)[0])
                events.append('v_{}'.format(velocity_value))
                events.append('d_{}'.format(note.duration))
                events.append('p_{}'.format(note.pitch))

        # create bar event
        events.append("|")

    # End of piece event
    events.append("e")

    return events

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

def _get_closest_note_value(delta_time, beat_resol, tempo):
    note_types = _get_duration_values(beat_resol=beat_resol, tempo=tempo)

    min_dist = math.inf
    min_type = None
    for i, type in enumerate(note_types):
        dist = abs(delta_time - type)
        if dist < min_dist:
            min_dist = dist
            min_type = i

    return min_type

def encode_midi(path_infile, path_outfile, note_sorting=1):
    # --- load --- #
    midi_obj = miditoolkit.midi.parser.MidiFile(path_infile)

    beat_resol = midi_obj.ticks_per_beat
    bar_resol  = beat_resol * 4
    tick_resol = beat_resol // 4

    # notes and tempo changes
    notes = _load_notes(midi_obj, note_sorting=note_sorting)
    tempo_changes = _load_tempo_changes(midi_obj)
    chords = _load_chords(midi_obj, beat_resol)

    # --- process items to grid --- #
    # compute empty bar offset at head
    first_note_time = min([notes[k][0].start for k in notes.keys()])
    last_note_time = max([notes[k][-1].start for k in notes.keys()])

    # compute quantized time of the first note
    quant_time_first = int(np.round(first_note_time / tick_resol) * tick_resol)

    # compute quantized offset and last bar time
    offset = quant_time_first // bar_resol
    last_bar = int(np.ceil(last_note_time / bar_resol)) - offset
    print(' > offset:', offset)
    print(' > last_bar:', last_bar)

    # process quantized notes and tempo
    note_grid = _process_notes(notes, tempo_changes, offset, tick_resol, beat_resol, bar_resol)
    tempo_grid = _process_tempo_changes(tempo_changes, offset, tick_resol, bar_resol)
    chord_grid = _process_chords(chords, offset, tick_resol, bar_resol)

    # create events
    events = _create_events(note_grid, tempo_grid, chord_grid, last_bar, tick_resol, bar_resol)

    # save
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)

    with open(path_outfile, 'w') as f:
        f.write(' '.join([str(e) for e in events]))

    return events

def decode_midi(idx_array, path_outfile=None, ticks_per_beat=1024):
    # Create mid object
    midi_obj = miditoolkit.midi.parser.MidiFile(ticks_per_beat=ticks_per_beat)

    beat_resol = ticks_per_beat
    bar_resol  = beat_resol * 4
    tick_resol = beat_resol // 4

    events_index = Event.build_events_index(bar_resol, tick_resol)
    events = [Event.from_int(idx, events_index) for idx in idx_array]

    bar_cnt = 0
    cur_pos = 0

    notes = []
    for ev in events:
        if ev.type == "beat":
            cur_pos = bar_cnt * bar_resol + ev.value * tick_resol

        if ev.type == "tempo":
            tempo = DEFAULT_TEMPO_BINS[ev.value]
            midi_obj.tempo_changes.append(TempoChange(tempo=tempo, time=cur_pos))

        if ev.type == "velocity":
            velocity = DEFAULT_VELOCITY_BINS[ev.value]

        if ev.type == "duration":
            note_values = _get_duration_values(beat_resol=beat_resol, tempo=tempo)
            duration = note_values[ev.value]

        if ev.type == "pitch":
            note = Note(pitch=ev.value, start=cur_pos, end=cur_pos + duration, velocity=velocity)
            notes.append(note)

        elif ev.type == "bar":
            bar_cnt += 1

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
