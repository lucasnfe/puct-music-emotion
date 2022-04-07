import os
import pickle
import argparse
import numpy as np
import pandas as pd

from utils import *

# config
BEAT_RESOL = 1024 # ticks per beat
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

# ---- define event ---- # #todo
''' 8 kinds:
     tempo: 0:   IGN
            1:   no change
            int: tempo
     chord: 0:   IGN
            1:   no change
            str: chord types
  bar-beat: 0:   IGN
            int: beat position (1...16)
            int: bar (bar)
      type: 0:   eos
            1:   metrical
            2:   note
            3:   emotion
  duration: 0:   IGN
            int: length
     pitch: 0:   IGN
            int: pitch
  velocity: 0:   IGN
            int: velocity
  emotion:  0:   IGN
            1:   Q1
            2:   Q2
            3:   Q3
            4:   Q4
'''

# event template
compound_event = {
    'tempo': 0,
    'chord': 0,
    'bar-beat': 0,
    'type': 0,
    'pitch': 0,
    'duration': 0,
    'velocity': 0,
    'emotion': 0
}

def create_emo_event(emo_tag):
    emo_event = compound_event.copy()
    emo_event['emotion'] = emo_tag
    emo_event['type'] = 'Emotion'
    return emo_event

def create_bar_event():
    meter_event = compound_event.copy()
    meter_event['bar-beat'] = 'Bar'
    meter_event['type'] = 'Metrical'
    return meter_event

def create_piano_metrical_event(tempo, chord, pos):
    meter_event = compound_event.copy()
    meter_event['tempo'] = tempo
    meter_event['chord'] = chord
    meter_event['bar-beat'] = pos
    #todo
    meter_event['type'] = 'Metrical'
    return meter_event

def create_piano_note_event(pitch, duration, velocity):
    note_event = compound_event.copy()
    note_event['pitch'] = pitch
    note_event['duration'] = duration
    note_event['velocity'] = velocity
    note_event['type'] = 'Note'
    return note_event

def create_eos_event():
    eos_event = compound_event.copy()
    eos_event['type'] = 'EOS'
    return eos_event

# ----------------------------------------------- #
# core functions
def corpus2event_cp(path_infile, path_outfile):
    '''
    task: 2 track
        1: piano      (note + tempo)
    ---
    remove duplicate position tokens
    '''

    data = pickle.load(open(path_infile, 'rb'))

    # global tag
    global_end = data['metadata']['last_bar'] * BAR_RESOL

    # emotion tag
    emo_tag = 0
    if data['metadata']['emotion'] in emo_map:
        emo_tag = emo_map[data['metadata']['emotion']]

    # process
    final_sequence = []
    final_sequence.append(create_emo_event(emo_tag))
    for bar_step in range(0, global_end, BAR_RESOL):
        final_sequence.append(create_bar_event())

        # --- piano track --- #
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            pos_on = False
            pos_events = []
            pos_text = 'Beat_' + str((timing-bar_step)//TICK_RESOL)

            # unpack
            t_chords = data['chords'][timing]
            t_tempos = data['tempos'][timing]
            t_notes = data['notes'][0][timing] # piano track

            # metrical
            #todo
            if len(t_tempos) or len(t_chords):
                # chord
                if len(t_chords):
                    root, quality, bass = t_chords[-1].text.split('_')
                    chord_text = '{}_{}'.format(root, quality)
                else:
                    chord_text = 'CONTI'

                # tempo
                if len(t_tempos):
                    tempo_text = 'Tempo_' + str(t_tempos[-1].tempo)
                else:
                    tempo_text = 'CONTI'

                # create
                pos_events.append(
                    create_piano_metrical_event(
                        tempo_text, chord_text, pos_text))
                pos_on = True

            # note
            if len(t_notes):
                if not pos_on:
                    pos_events.append(
                        create_piano_metrical_event(
                            'CONTI', 'CONTI', pos_text))

                for note in t_notes:
                    note_pitch_text = 'Note_Pitch_' + str(note.pitch)
                    note_duration_text = 'Note_Duration_' + str(note.duration)
                    note_velocity_text = 'Note_Velocity_' + str(note.velocity)

                    pos_events.append(
                        create_piano_note_event(
                            note_pitch_text,
                            note_duration_text,
                            note_velocity_text))

            # collect & beat
            if len(pos_events):
                final_sequence.extend(pos_events)

    # BAR ending
    final_sequence.append(create_bar_event())

    # EOS
    final_sequence.append(create_eos_event())

    # save
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
    pickle.dump(final_sequence, open(path_outfile, 'wb'))

    return len(final_sequence)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='corpus2events.py')
    parser.add_argument('--path_indir', type=str, required=True)
    parser.add_argument('--path_outdir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(
        args.path_indir,
        extension=('pkl'),
        is_pure=True,
        is_sort=True)
    n_files = len(midifiles)
    print('num files:', n_files)

    # run all
    len_list = []
    paths = []
    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        print('{}/{}'.format(fidx+1, n_files))

        # paths
        path_infile = os.path.join(args.path_indir, path_midi)
        path_outfile = os.path.join(args.path_outdir, path_midi)

        # proc
        num_tokens = corpus2event_cp(path_infile, path_outfile)
        print(' > num_token:', num_tokens)

        len_list.append(num_tokens)
        paths.append(path_midi)

    plot_hist(len_list, os.path.join(args.path_outdir, 'num_tokens.png'))

    d = {'filename': paths, 'num_tokens': len_list}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(args.path_outdir, 'len_token.csv'), index=False)
