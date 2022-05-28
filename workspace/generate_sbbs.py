import os
import copy
import json
import torch
import argparse
import numpy as np

from mcts import MCTS

from generate import * 
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

def is_terminal(beam, n_bars, seq_len):
    is_finished = torch.sum(beam == BAR_TOKEN, dim=-1) >= n_bars
    is_oversized = beam.shape[-1] >= seq_len
    return is_finished.any() or is_oversized

def generate(language_model, emotion_classifier, emotion, n_bars, seq_len, vocab_size, piece, k=0, b=10, t=1.0):
    try:
        print("Current piece:", piece)
        init_tokens = prime.unsqueeze(dim=0)

        # Get probabilities of next token
        y_i = language_model(init_tokens)[:,-1,:]

        # Filter out end token
        y_i = filter_index(y_i, END_TOKEN)
   
        # Filter top p
        if k > 0:
            y_i = filter_top_k(y_i, k)
        y_i = torch.softmax(y_i, dim=1)
       
        # Batchfy prime sequence
        first_top_ixs = torch.where(y_i > 0)[1]
        first_top_ps = y_i[:,first_top_ixs]
        
        first_beam = init_tokens.repeat(k, 1)
        first_top_ixs = torch.reshape(first_top_ixs, (k, 1))
        first_top_ps = torch.reshape(first_top_ps, (k, 1))
        
        first_beam = torch.cat((first_beam, first_top_ixs), axis=1)

        # Compute emotion score
        emotion_scores = torch.softmax(emotion_classifier(first_beam), dim=1)[:,emotion]
        emotion_scores = torch.reshape(emotion_scores, [k, 1])

        combined_scores = torch.log(first_top_ps) + torch.log(emotion_scores)

        # Sample first beam
        ps = torch.softmax(torch.reshape(combined_scores, (-1,)), dim=0)
        first_beam_ixs = torch.multinomial(ps, num_samples=b)

        beam_scores = torch.log(first_top_ps[first_beam_ixs])
        beam_seqs = first_beam[first_beam_ixs]

        print(beam_scores)
        print(beam_seqs)

        while not is_terminal(beam_seqs, n_bars, seq_len):
            beam_candidates = []
            beam_candidates_ps = []
            beam_candidates_scores = []
            
            # Loop over each beam to address GPU memory issues
            for b_seq, b_score in zip(beam_seqs, beam_scores):
                b_seq = b_seq.unsqueeze(0)

                y_i = language_model(b_seq)[:,-1,:]
                y_i = filter_index(y_i, END_TOKEN)
         
                # Filter top p
                if k > 0:
                    y_i = filter_top_k(y_i, k)
                y_i = torch.softmax(y_i, dim=1)

                b_top_ixs = torch.where(y_i > 0)[1]
                b_top_ps = y_i[:, b_top_ixs]

                b_candidates = b_seq.repeat(k, 1)
                b_candidates_scores = b_score.repeat(k, 1)

                b_top_ixs = torch.reshape(b_top_ixs, (k, 1))
                b_top_ps = torch.reshape(b_top_ps, (k, 1))

                b_candidates = torch.cat((b_candidates, b_top_ixs), axis=1)

                # Compute emotion scores
                b_emotion_scores = torch.softmax(emotion_classifier(b_candidates), dim=1)[:,emotion]
                b_emotion_scores = torch.reshape(b_emotion_scores, [k, 1])
   
                b_combined_scores = b_candidates_scores + torch.log(b_top_ps) + torch.log(b_emotion_scores)

                beam_candidates.append(b_candidates)
                beam_candidates_ps.append(b_top_ps)
                beam_candidates_scores.append(b_combined_scores)
           
            beam_candidates = torch.cat(beam_candidates)
            beam_candidates_ps = torch.cat(beam_candidates_ps)
            beam_candidates_scores = torch.cat(beam_candidates_scores)

            # Sample first beam
            ps = torch.softmax(torch.reshape(beam_candidates_scores, (-1,)), dim=0)
            next_beam_ixs = torch.multinomial(ps, num_samples=b)

            beam_scores = beam_scores + torch.log(beam_candidates_ps[next_beam_ixs])
            beam_seqs = beam_candidates[next_beam_ixs]
            
            print(beam_scores)
            print(beam_seqs)
    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt.")

    piece_ix = torch.argmax(beam_candidates_scores)
    piece = beam_candidates[piece_ix]
    
    return piece.cpu().numpy().tolist()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='genrate_mcts.py')
    parser.add_argument('--lm', type=str, required=True, help="Path to load language model from.")
    parser.add_argument('--clf', type=str, required=True, help="Path to load emotion classifier from.")
    parser.add_argument('--emotion', type=int, required=True, help="Piece emotion.")
    parser.add_argument('--k', type=int, default=0.0, help="Number k of elements to consider while sampling.")
    parser.add_argument('--b', type=int, default=10, help="Beam size.")
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
    
    # Define prime sequence
    prime = [Event(event_type='control', value=0).to_int(),
             Event(event_type='emotion', value=0).to_int(),
             Event(event_type='beat', value=0).to_int()]
    
    with torch.no_grad():
        #prime = torch.tensor(prime).unsqueeze(dim=0).to(device)
        prime = torch.tensor(prime, device=device)

        # Generate piece with mcts
        print('> Starting to generate with SBBS')
        print('-' * 50)
        print('Parameters:')
        print(f'k: {opt.k}')
        print(f'b: {opt.b}')
        print(f'Emotion: {opt.emotion}')
        print(f'Number of bars: {opt.n_bars}')
        print('-' * 50)

        piece = generate(language_model, emotion_classifier, opt.emotion, opt.n_bars, opt.seq_len, VOCAB_SIZE, prime, k=opt.k, b=opt.b, t=opt.t)
        decode_midi(piece, opt.save_to)
        print(piece)

