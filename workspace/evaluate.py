import os
import muspy
import argparse
import torch
import numpy as np

from encoder import *
from generate_mcts import load_classifier

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
    parser.add_argument('--device', type=str, required=False, help="Force device.")
    args = parser.parse_args()

    # Disable autograd globally
    torch.autograd.set_grad_enabled(False)

    # Set up torch device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load emotion classifier
    emotion_classifier = load_classifier(args.clf, VOCAB_SIZE, args.d_model, args.n_layers, args.n_heads, args.seq_len, out_size=4, device=device)
    print(f'> Loaded emotion classifier {args.clf}')

    discriminator = load_classifier(args.disc, VOCAB_SIZE, args.d_model, args.n_layers, args.n_heads, args.seq_len, out_size=1, device=device)
    print(f'> Loaded discriminator {args.disc}')

    # list files
    midifiles = traverse_dir(
        args.path_indir,
        is_pure=True,
        is_sort=True)
    n_files = len(midifiles)
    print('num files:', n_files)

    # Define metrics
    metrics = {}

    emotion_hits = []
    discriminator_hits = []

    for fidx in range(n_files):
        path_midi = midifiles[fidx]
        path_infile = os.path.join(args.path_indir, path_midi)

        print('---- File {}: {}'.format(fidx, path_midi))

        #  Load muspy object
        midi = muspy.read_midi(path_infile)

        # Pitch related metrics
        #pitch_range.append(muspy.pitch_range(midi))
        #n_pitches_used.append(muspy.n_pitch_classes_used(midi))
        #polyphony.append(muspy.polyphony(midi))

        # Emotion
        emotion_target = process_emotion(path_midi) - 1

        if emotion_target not in metrics:
            metrics[emotion_target] = {'PR': [], 'NPC': [], 'POLY': [], 'EMOTION': [], 'DISC': []}

        metrics[emotion_target]['PR'].append(muspy.pitch_range(midi))
        metrics[emotion_target]['NPC'].append(muspy.n_pitch_classes_used(midi))
        metrics[emotion_target]['POLY'].append(muspy.polyphony(midi))

        x = torch.tensor(encode_midi(path_infile)[:args.seq_len], device=device).unsqueeze(0)
        y = torch.softmax(emotion_classifier(x), dim=1).squeeze()
        
        emotion_hat = int(torch.argmax(y))
        metrics[emotion_target]['EMOTION'].append(int(emotion_hat == emotion_target))

        # Discriminator score
        discriminator_hat = float(torch.round(torch.sigmoid(discriminator(x))).squeeze())
        metrics[emotion_target]['DISC'].append(int(discriminator_hat == 1.0))

    print(discriminator_hits)

    print("Evaluation")
    for emotion in metrics:
        print('- Emotion', emotion)
        print("> PR: {:.2f}/{:.2f}".format(np.mean(metrics[emotion]['PR']), np.std(metrics[emotion]['PR'])))
        print("> NPC: {:.2f}/{:.2f}".format(np.mean(metrics[emotion]['NPC']), np.std(metrics[emotion]['NPC'])))
        print("> POLY: {:.2f}/{:.2f}".format(np.mean(metrics[emotion]['POLY']), np.std(metrics[emotion]['POLY'])))

        print("> Emotion: {:.2f}".format(np.mean(metrics[emotion]['EMOTION'])))
        print("> Discriminator: {:.2f}".format(np.mean(metrics[emotion]['DISC'])))
