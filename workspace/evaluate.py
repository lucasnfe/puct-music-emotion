import os
import muspy
import argparse
import torch
import numpy as np

from encoder import process_emotion
from generate_mcts import load_classifier,

def traverse_dir(
        root_dir,
        extension=('mid', 'MID', 'midi'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=False,
        is_sort=False,
        is_ext=True):

    if verbose:
        print('[*] Scanning...')

    cnt, file_list = 0, []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue

                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')

    if is_sort:
        file_list.sort()

    return file_list

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='evaluate.py')
    parser.add_argument('--path_indir', type=str, required=True)
    parser.add_argument('--clf', type=str, required=True)
    parser.add_argument('--disc', type=str, required=True)
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--n_layers', type=int, default=8, help="Number of transformer layers.")
    parser.add_argument('--d_model', type=int, default=512, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    args = parser.parse_args()

    # Load emotion classifier
    emotion_classifier = load_classifier(args.clf, vocab_size, args.d_model, args.n_layers, args.n_heads, args.seq_len, out_size=4)
    print(f'> Loaded emotion classifier {args.clf}')

    discriminator = load_classifier(args.disc, vocab_size, args.d_model, args.n_layers, args.n_heads, args.seq_len, out_size=1)
    print(f'> Loaded discriminator {args.disc}')

    # list files
    midifiles = traverse_dir(
        args.path_indir,
        is_pure=True,
        is_sort=True)
    n_files = len(midifiles)
    print('num files:', n_files)

    # Define metrics
    pitch_range = []
    n_pitches_used = []
    polyphony = []

    emotion_hits = []
    discriminator_scores = []

    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        path_infile = os.path.join(args.path_indir, path_midi)

        print('---- File {}: {}'.format(fidx, path_midi))

        #  Load muspy object
        midi = muspy.read_midi(path_infile)

        # Pitch related metrics
        pitch_range.append(muspy.pitch_range(midi))
        n_pitches_used.append(muspy.n_pitches_used(midi))
        polyphony.append(muspy.polyphony(midi))

        # Emotion
        emotion_target = process_emotion(path_midi)

        y_hat = torch.softmax(emotion_classifier(roll_state), dim=1).squeeze()
        emotion_hat = int(torch.argmax(y_hat))

        emotion_hits.append(int(emotion_hat == emotion_target))

        # Discriminator score
        y_hat = discriminator(roll_state)
        discriminator_score = torch.sigmoid(y_hat).squeeze()
        discriminator_scores.append(discriminator_score)

    print("Evaluation")
    print("> PR: {}/{}".format(np.mean(pitch_range), np.std(pitch_range)))
    print("> NPC: {}/{}".format(np.mean(n_pitches_used), np.std(n_pitches_used)))
    print("> POLY: {}/{}".format(np.mean(polyphony), np.std(polyphony)))

    print("> Emotion: {}/{}".format(np.mean(emotion_hits)))
    print("> Discriminator: {}/{}".format(np.mean(discriminator_scores), np.std(discriminator_scores)))
