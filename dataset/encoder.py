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
import argparse
import pretty_midi
import multiprocessing as mp

from utils import traverse_dir

RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100
RANGE_SPECIAL = 2

START_IDX = {
    'note_on': 0,
    'note_off': RANGE_NOTE_ON,
    'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
    'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT,
    'special': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL
}

MIDI_EXTENSIONS = [".mid", ".midi"]

class SustainAdapter:
    def __init__(self, time, type):
        self.start =  time
        self.type = type

class SustainDownManager:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.managed_notes = []
        self._note_dict = {} # key: pitch, value: note.start

    def add_managed_note(self, note: pretty_midi.Note):
        self.managed_notes.append(note)

    def transposition_notes(self):
        for note in reversed(self.managed_notes):
            try:
                note.end = self._note_dict[note.pitch]
            except KeyError:
                note.end = max(self.end, note.end)

            self._note_dict[note.pitch] = note.start

# Divided note by note_on, note_off
class SplitNote:
    def __init__(self, type, time, value, velocity):
        # type: note_on, note_off
        self.type = type
        self.time = time
        self.velocity = velocity
        self.value = value

    def __repr__(self):
        return '<[SNote] time: {} type: {}, value: {}, velocity: {}>'\
            .format(self.time, self.type, self.value, self.velocity)

class Event:
    def __init__(self, event_type, value):
        self.type = event_type
        self.value = value

    def __repr__(self):
        return '<Event type: {}, value: {}>'.format(self.type, self.value)

    def to_int(self):
        return START_IDX[self.type] + self.value

    @staticmethod
    def from_int(int_value):
        info = Event._type_check(int_value)
        return Event(info['type'], info['value'])

    @staticmethod
    def _type_check(int_value):
        range_note_on = range(0, RANGE_NOTE_ON)
        range_note_off = range(RANGE_NOTE_ON, RANGE_NOTE_ON+RANGE_NOTE_OFF)
        range_time_shift = range(RANGE_NOTE_ON+RANGE_NOTE_OFF, RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT)
        range_velocity = range(RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT, RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT+RANGE_VEL)

        valid_value = int_value

        if int_value in range_note_on:
            return {'type': 'note_on', 'value': valid_value}
        elif int_value in range_note_off:
            valid_value -= RANGE_NOTE_ON
            return {'type': 'note_off', 'value': valid_value}
        elif int_value in range_time_shift:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF)
            return {'type': 'time_shift', 'value': valid_value}
        elif int_value in range_velocity:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)
            return {'type': 'velocity', 'value': valid_value}

        valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT + RANGE_VEL)
        return {'type': 'special', 'value': valid_value}

def _divide_note(notes):
    result_array = []
    notes.sort(key=lambda x: x.start)

    for note in notes:
        on = SplitNote('note_on', note.start, note.pitch, note.velocity)
        off = SplitNote('note_off', note.end, note.pitch, None)
        result_array += [on, off]

    return result_array

def _merge_note(snote_sequence):
    note_on_dict = {}

    result_array = []
    for snote in snote_sequence:
        if snote.type == 'note_on':
            note_on_dict[snote.value] = snote
        elif snote.type == 'note_off':
            try:
                # Get associated note_on time
                on = note_on_dict[snote.value]

                # Get note_off time
                off = snote
                if off.time - on.time == 0:
                    continue

                # Create pretty_midi note from the note_on and note_off times
                result = pretty_midi.Note(on.velocity, snote.value, on.time, off.time)
                result_array.append(result)
            except:
                print('info removed pitch: {}'.format(snote.value))

    return result_array

def _snote2events(snote: SplitNote, prev_vel: int):
    result = []
    if snote.velocity is not None:
        modified_velocity = snote.velocity // 4
        if prev_vel != modified_velocity:
            result.append(Event(event_type='velocity', value=modified_velocity))

    result.append(Event(event_type=snote.type, value=snote.value))
    return result

def _event_seq2snote_seq(event_sequence):
    timeline = 0
    velocity = 0
    snote_seq = []

    for event in event_sequence:
        if event.type == 'time_shift':
            timeline += ((event.value+1) / 100)
        if event.type == 'velocity':
            velocity = event.value * 4
        if event.type == 'special':
            continue
        else:
            snote = SplitNote(event.type, timeline, event.value, velocity)
            snote_seq.append(snote)

    return snote_seq

def _make_time_sift_events(prev_time, post_time):
    time_interval = int(round((post_time - prev_time) * 100))
    results = []

    while time_interval >= RANGE_TIME_SHIFT:
        results.append(Event(event_type='time_shift', value=RANGE_TIME_SHIFT-1))
        time_interval -= RANGE_TIME_SHIFT

    if time_interval == 0:
        return results

    return results + [Event(event_type='time_shift', value=time_interval-1)]

def _control_preprocess(ctrl_changes):
    sustains = []

    manager = None
    for ctrl in ctrl_changes:
        # sustain down
        if ctrl.value >= 64 and manager is None:
            manager = SustainDownManager(start=ctrl.time, end=None)
        # sustain up
        elif ctrl.value < 64 and manager is not None:
            manager.end = ctrl.time
            sustains.append(manager)
            manager = None
        elif ctrl.value < 64 and len(sustains) > 0:
            sustains[-1].end = ctrl.time

    return sustains

def _note_preprocess(susteins, notes):
    note_stream = []

    # if the midi file has sustain controls
    if susteins:
        for sustain in susteins:
            for note_idx, note in enumerate(notes):
                if note.start < sustain.start:
                    note_stream.append(note)
                elif note.start > sustain.end:
                    notes = notes[note_idx:]
                    sustain.transposition_notes()
                    break
                else:
                    sustain.add_managed_note(note)

        for sustain in susteins:
            note_stream += sustain.managed_notes

    # else, just push everything into note stream
    else:
        for note_idx, note in enumerate(notes):
            note_stream.append(note)

    note_stream.sort(key= lambda x: x.start)
    return note_stream

def encode_midi(path_infile, path_outfile):
    print('----')
    print(' >', path_infile)
    print(' >', path_outfile)

    notes = []
    events = []

    mid = pretty_midi.PrettyMIDI(midi_file=path_infile)
    for inst in mid.instruments:
        # Only consider instruments from the piano family
        assert pretty_midi.program_to_instrument_class(inst.program) == "Piano", file_path + " contains a non-piano instrument."

        # ctrl.number is the number of sustain control:
        # https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
        ctrls = _control_preprocess([ctrl for ctrl in inst.control_changes if ctrl.number == 64])
        notes += _note_preprocess(ctrls, inst.notes)

    dnotes = _divide_note(notes)
    dnotes.sort(key=lambda x: x.time)

    cur_time = 0
    cur_vel = 0
    for snote in dnotes:
        events += _make_time_sift_events(prev_time=cur_time, post_time=snote.time)
        events += _snote2events(snote=snote, prev_vel=cur_vel)

        cur_time = snote.time
        cur_vel = snote.velocity

    # Make sure the encoded midi is not empty
    assert len(events) > 0, file_path + " does not have any MIDI events."

    # Create START and END events
    start = Event(event_type="special", value=0)
    end = Event(event_type="special", value=1)

    # add START and END events
    events = [start.to_int()] + [e.to_int() for e in events] + [end.to_int()]

    # mkdir
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)

    # Save txt version of the midi file to load it faster during training
    with open(path_outfile, "w") as midi_txt_file:
        midi_txt_file.write(" ".join(str(e) for e in events))

def decode_midi(idx_array, file_path=None):
    event_sequence = [Event.from_int(idx) for idx in idx_array]

    snote_seq = _event_seq2snote_seq(event_sequence)
    note_seq = _merge_note(snote_seq)
    note_seq.sort(key=lambda x:x.start)

    # if want to change instument, see https://www.midi.org/specifications/item/gm-level-1-sound-set
    mid = pretty_midi.PrettyMIDI()
    instument = pretty_midi.Instrument(1, False, "Encoded Midi")
    instument.notes = note_seq

    mid.instruments.append(instument)
    if file_path is not None:
        mid.write(file_path)

    return mid

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
