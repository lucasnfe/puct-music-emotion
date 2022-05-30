#
# Generate MIDI piano pieces with fast transformer.
#
# Author: Lucas N. Ferreira - lucasnfe@gmail.com
#
#

import os
import torch
import json
import math
import argparse

from encoder import *
from model import MusicGenerator

from torch.distributions.categorical import Categorical

END_TOKEN = Event(event_type='control', value=2).to_int()
BAR_TOKEN = Event(event_type='control', value=1).to_int()

def is_terminal(state, n_bars, seq_len):
    if len(state) == 0:
        return False
    
    if len(state) >= seq_len:
        return True
    
    b = 0
    for t in state:
        if t == BAR_TOKEN:
            b += 1

    if b >= n_bars:
        return True

    if state[-1] == END_TOKEN:
        return True

    return False

def generate(prime, n_bars, seq_len):
    # Generate new tokens
    generated = []

    i = 0
    while not is_terminal(generated, n_bars, seq_len):
        generated.append(prime[i])
        i += 1

    last_bar = len(generated) - generated[::-1].index(BAR_TOKEN) - 1

    return [int(token) for token in generated[:last_bar + 1]]

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='generate.py')
    parser.add_argument('--prime', type=str, required=True, help="Prime sequence.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--n_bars', type=int, default=4, help="Num bars to generate.")
    parser.add_argument('--save_to', type=str, required=True, help="Directory to save the generated samples.")
    opt = parser.parse_args()

    # Encode prime file
    prime = encode_midi(opt.prime)

    # Generate piece
    piece = generate(prime, n_bars=opt.n_bars, seq_len=opt.seq_len)
    decode_midi(piece, opt.save_to)
    print(piece)
