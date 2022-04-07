import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import *

# config
BEAT_RESOL = 1024
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4

# define event
def create_event(name, value):
    event = dict()
    event['name'] = name
    event['value'] = value
    return event

# core functions
def corpus2event_remi_v2(path_infile, path_outfile):
    '''
    <<< REMI v2 >>>
    task: 2 track
       1: piano      (note + tempo + chord)
    ---
    remove duplicate position tokens
    '''
    data = pickle.load(open(path_infile, 'rb'))

    # global tag
    global_end = data['metadata']['last_bar'] * BAR_RESOL

    # process
    final_sequence = []
    for bar_step in range(0, global_end, BAR_RESOL):
        final_sequence.append(create_event('Bar', None))

        # --- piano track --- #
        for timing in range(bar_step, bar_step + BAR_RESOL, TICK_RESOL):
            pos_events = []

            # unpack
            t_chords = data['chords'][timing]
            t_tempos = data['tempos'][timing]
            t_notes = data['notes'][0][timing] # piano track

            # chord
            if len(t_chords):
                root, quality, bass = t_chords[0].text.split('_')
                pos_events.append(create_event('Chord', '{}_{}'.format(root, quality)))

            # tempo
            if len(t_tempos):
                pos_events.append(create_event('Tempo', t_tempos[0].tempo))

            # note
            if len(t_notes):
                for note in t_notes:
                    pos_events.extend([
                        create_event('Note_Pitch', note.pitch),
                        create_event('Note_Velocity', note.velocity),
                        create_event('Note_Duration', note.duration),
                    ])

            # collect & beat
            if len(pos_events):
                final_sequence.append(
                    create_event('Beat', (timing-bar_step)//TICK_RESOL))
                final_sequence.extend(pos_events)

    # BAR ending
    final_sequence.append(create_event('Bar', None))

    # EOS
    final_sequence.append(create_event('EOS', None))

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
    print('num fiels:', n_files)

    # run all
    len_list = []
    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        print('{}/{}'.format(fidx, n_files))

        # paths
        path_infile = os.path.join(args.path_indir, path_midi)
        path_outfile = os.path.join(args.path_outdir, path_midi)

        # proc
        num_tokens = corpus2event_remi_v2(path_infile, path_outfile)
        print(' > num_token:', num_tokens)
        len_list.append(num_tokens)

    # plot
    plot_hist(len_list, os.path.join(args.path_outdir, 'num_tokens.png'))
