import os
import copy
import argparse
import numpy as np
import multiprocessing as mp

import miditoolkit
from miditoolkit.midi import parser as mid_parser
from miditoolkit.pianoroll import parser as pr_parser
from miditoolkit.midi.containers import Marker, Instrument, TempoChange

from chorder import Dechorder
from utils import traverse_dir

num2pitch = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
}

BEAT_RESOL = 1024 # ticks per beat

def proc_one(path_infile, path_outfile):
    print('----')
    print(' >', path_infile)
    print(' >', path_outfile)

    # load
    midi_obj = miditoolkit.midi.parser.MidiFile(path_infile)
    midi_obj_out = copy.deepcopy(midi_obj)
    notes = midi_obj.instruments[0].notes
    notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    # --- chord --- #
    # exctract chord
    chords = Dechorder.dechord(midi_obj)

    markers = []
    for cidx, chord in enumerate(chords):
        if chord.is_complete():
            chord_text = '{}_{}_{}'.format(num2pitch[chord.root_pc], chord.quality, num2pitch[chord.bass_pc])
        else:
            chord_text = 'N_N_N'
        markers.append(Marker(time=int(cidx*BEAT_RESOL), text=chord_text))

    # dedup
    prev_chord = None
    dedup_chords = []
    for m in markers:
        if m.text != prev_chord:
            prev_chord = m.text
            dedup_chords.append(m)

    # --- global properties --- #
    # global tempo
    tempos = [b.tempo for b in midi_obj.tempo_changes][:40]
    tempo_median = np.median(tempos)
    global_bpm =int(tempo_median)
    print(' > [global] bpm:', global_bpm)

    # === save === #
    # mkdir
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)

    # markers
    midi_obj_out.markers = dedup_chords
    midi_obj_out.markers.insert(0, Marker(text='global_bpm_'+str(int(global_bpm)), time=0))

    # save
    midi_obj_out.instruments[0].name = 'piano'
    midi_obj_out.dump(path_outfile)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='midi_encoder.py')
    parser.add_argument('--path_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    args = parser.parse_args()

    # paths
    os.makedirs(args.path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(
        args.path_indir,
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
        path_infile = os.path.join(args.path_indir, path_midi)
        path_outfile = os.path.join(args.path_outdir, path_midi)

        # append
        data.append([path_infile, path_outfile])

    # run, multi-thread
    pool = mp.Pool()
    pool.starmap(proc_one, data)
