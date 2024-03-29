import os
import copy
import json
import torch
import argparse
import numpy as np

from mcts import MCTS

from encoder import *
from model import *

def load_language_model(model, vocab_size, d_model, n_layers, n_heads, seq_len, device):
    language_model = MusicGenerator(n_tokens=vocab_size,
                            d_model=d_model,
                            seq_len=seq_len,
                     attention_type="causal-linear",
                           n_layers=n_layers,
                           n_heads=n_heads).to(device)

    # Load model
    language_model.load_state_dict(torch.load(model, map_location=device)["model_state"])
    language_model.eval()

    return language_model

def load_classifier(model, vocab_size, d_model, n_layers, n_heads, seq_len, out_size, device):
    # Load Emotion Classifier
    emotion_classifier = MusicClassifier(n_tokens=vocab_size,
                            d_model=d_model,
                            seq_len=seq_len,
                     attention_type="linear",
                           n_layers=n_layers,
                           n_heads=n_heads).to(device)

   # Add classification head
    emotion_classifier = torch.nn.Sequential(emotion_classifier,
                                torch.nn.Dropout(0.0),
                                torch.nn.Linear(vocab_size, out_size)).to(device)


    emotion_classifier.load_state_dict(torch.load(model, map_location=device)["model_state"])
    emotion_classifier.eval()

    return emotion_classifier

def generate(language_model, emotion_classifier, discriminator, emotion, n_bars, seq_len, vocab_size, piece, roll_steps=30, p=0, c=1.0, t=1.0):
    tree = MCTS(language_model,
                emotion_classifier,
                discriminator,
                emotion,
                vocab_size,
                device,
                n_bars,
                seq_len,
                p, c)

    # Init mucts
    try:
        while True:
            print("Current piece:", piece)

            for step in range(roll_steps):
                print("Rollout: %d" % step)
                tree.step(piece)

            # Choose next state
            token = tree.choose(piece, temperature=t)
            piece = tree._get_next_state(piece, token)

            if tree._is_terminal(piece):
                break

    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt.")

    return piece.cpu().numpy().tolist()
    

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='genrate_mcts.py')
    parser.add_argument('--lm', type=str, required=True, help="Path to load language model from.")
    parser.add_argument('--clf', type=str, required=True, help="Path to load emotion classifier from.")
    parser.add_argument('--disc', type=str, required=True, help="Path to load discriminator from.")
    parser.add_argument('--emotion', type=int, required=True, help="Piece emotion.")
    parser.add_argument('--roll_steps', type=int, default=30, help="Number rollout steps.")
    parser.add_argument('--p', type=float, default=0.0, help="Number k of elements to consider while sampling.")
    parser.add_argument('--c', type=float, default=1.0, help="Constant c for puct.")
    parser.add_argument('--t', type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--n_bars', type=int, default=4, help="Num bars to generate.")
    parser.add_argument('--n_layers', type=int, default=8, help="Number of transformer layers.")
    parser.add_argument('--d_model', type=int, default=512, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--save_to', type=str, required=True, help="Set a file to save the models to.")
    parser.add_argument('--device', type=str, required=False, help="Force device.")
    opt = parser.parse_args()

    # Disable autograd globally
    torch.autograd.set_grad_enabled(False)

    # Set up torch device
    if opt.device:
        device = torch.device(opt.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pad_token = Event(event_type='control', value=3).to_int()

    # Load language models
    language_model = load_language_model(opt.lm, VOCAB_SIZE, opt.d_model, opt.n_layers, opt.n_heads, opt.seq_len, device=device)
    print(f'> Loaded language model {opt.lm}')

    # Load emotion classifier
    emotion_classifier = load_classifier(opt.clf, VOCAB_SIZE, opt.d_model, opt.n_layers, opt.n_heads, opt.seq_len, out_size=4, device=device)
    print(f'> Loaded emotion classifier {opt.clf}')
    
    discriminator = load_classifier(opt.disc, VOCAB_SIZE, opt.d_model, opt.n_layers, opt.n_heads, opt.seq_len, out_size=1, device=device)
    print(f'> Loaded discriminator {opt.disc}')

    # Define prime sequence
    prime = [Event(event_type='control', value=0).to_int(),
             Event(event_type='emotion', value=0).to_int(),
             Event(event_type='beat', value=0).to_int()]
    
    with torch.no_grad():
        #prime = torch.tensor(prime).unsqueeze(dim=0).to(device)
        prime = torch.tensor(prime, device=device)

        # Generate piece with mcts
        print('> Starting to generate with MCTS')
        print('-' * 50)
        print('Parameters:')
        print(f'p: {opt.p}')
        print(f'c: {opt.c}')
        print(f'Emotion: {opt.emotion}')
        print(f'Rollout steps: {opt.roll_steps}')
        print(f'Number of bars: {opt.n_bars}')
        print('-' * 50)

        piece = generate(language_model, emotion_classifier, discriminator, opt.emotion, opt.n_bars, opt.seq_len, VOCAB_SIZE, prime, opt.roll_steps, p=opt.p, c=opt.c, t=opt.t)
        decode_midi(piece, opt.save_to)
        print(piece)

