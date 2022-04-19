#
# Generate MIDI piano pieces with fast transformer.
#
# Author: Lucas N. Ferreira - lucasnfe@gmail.com
#
#

import torch
import json
import math
import argparse

from decoder import *
from model import MusicGenerator

from torch.distributions.categorical import Categorical

def filter_top_p(y_hat, p, filter_value=-float("Inf")):
    sorted_logits, sorted_indices = torch.sort(y_hat, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > p

    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    y_hat = y_hat.masked_fill(indices_to_remove, filter_value)

    return y_hat

def filter_top_k(y_hat, k, filter_value=-float("Inf")):
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = y_hat < torch.topk(y_hat, k)[0][..., -1, None]
    y_hat = y_hat.masked_fill(indices_to_remove, filter_value)

    return y_hat

def filter_repetition(previous_tokens, scores, penalty=1.0001):
    score = torch.gather(scores, 1, previous_tokens)

    # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
    score = torch.where(score < 0, score * penalty, score / penalty)

    scores.scatter_(1, previous_tokens, score)
    return scores

def sample_tokens(y_hat, num_samples=1):
    # Sample from filtered categorical distribution
    probs = torch.softmax(y_hat, dim=1)
    random_idx = torch.multinomial(probs, num_samples)
    return random_idx

def generate(model, prime, n, k=0, p=0, temperature=1.0):
    # Process prime sequence
    prime_len = len(prime)

    # Generate new tokens
    generated = torch.tensor(prime).unsqueeze(dim=0).to(device)
    for i in range(prime_len, prime_len + n):
        print("generated", generated)
        y_hat = model(generated)[:,-1,:]

        if k > 0:
            y_hat = filter_top_k(y_hat, k)
        if p > 0 and p < 1.0:
            y_hat = filter_top_p(y_hat, p)

        token = sample_tokens(y_hat)
        generated = torch.cat((generated, token), dim=1)

    return [int(token) for token in generated.squeeze(0)]

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='generate.py')
    parser.add_argument('--model', type=str, required=True, help="Path to load model from.")
    parser.add_argument('--vocab', type=str, required=True, help="Path to vocabulary.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--k', type=int, default=0, help="Number k of elements to consider while sampling.")
    parser.add_argument('--p', type=float, default=1.0, help="Probability p to consider while sampling.")
    parser.add_argument('--t', type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument('--n_layers', type=int, default=8, help="Number of transformer layers.")
    parser.add_argument('--d_model', type=int, default=256, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--beat_resol', type=int, default=1024, help="Ticks per beat.")
    parser.add_argument('--device', type=str, default=None, help="Torch device.")
    opt = parser.parse_args()

    # Set up torch device
    device = opt.device
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocabular
    with open(opt.vocab) as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    # Build linear transformer
    model = MusicGenerator(n_tokens=vocab_size,
                                     d_model=opt.d_model,
                                     seq_len=opt.seq_len,
                              attention_type="causal-linear",
                                    n_layers=opt.n_layers,
                                     n_heads=opt.n_heads).to(device)

    # Load model
    model.load_state_dict(torch.load(opt.model, map_location=device)["model_state"])
    model.eval()

    # Define prime sequence
    #prime = 's b_0'
    prime = 's b_0 t_33 c_A#_m v_46 d_20 p_82 v_44 d_4 p_46 b_1 v_44 d_27 p_77 b_2 v_44 d_27 p_89 b_3 v_44 d_27 p_77 b_4 c_A#_m v_44 d_27 p_87 b_5 v_44 d_26 p_77 b_6 v_44 d_26 p_85 b_7 v_44 d_27 p_77 b_8 c_A#_m v_45 d_27 p_84 b_9 t_32 v_44 d_26 p_77 b_10 v_44 d_26 p_85 b_11 t_32 v_44 d_27 p_77 b_12 c_A#_m v_44 d_27 p_84 b_13 v_44 d_26 p_77 b_14 t_32 v_44 d_26 p_82 b_15 v_44 d_27 p_77 |'
    prime = [vocab[event] for event in prime.split()]

    # Generate continuation
    # piece = generate_beam_search(model, prime, n=1000, beam_size=8, k=opt.k, p=opt.p, temperature=opt.t)
    piece = generate(model, prime, n=opt.seq_len - len(prime), k=opt.k, p=opt.p, temperature=opt.t)
    decode_midi(piece, vocab, "results/generated_piece.mid")
    print(piece)