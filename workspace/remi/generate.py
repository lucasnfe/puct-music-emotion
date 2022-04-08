#
# Generate MIDI piano pieces with fast transformer.
#
# Author: Lucas N. Ferreira - lucasnfe@gmail.com
#
#

import torch
import math
import pickle
import argparse

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

from torch.distributions.categorical import Categorical
from fast_transformers.builders import RecurrentEncoderBuilder

# ================================ #
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4
INSTR_NAME_MAP = {'piano': 0, 'melody': 1}

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, i):
        pos_embedding =  self.pe[0, i:i+1]
        x = torch.cat([x, pos_embedding.expand_as(x)], dim=1)
        return self.dropout(x)

class RecurrentMusicGenerator(torch.nn.Module):
    def __init__(self, n_tokens, d_model, seq_len,
                 attention_type="full", n_layers=4, n_heads=4,
                 d_query=32, dropout=0.0, softmax_temp=None,
                 attention_dropout=0.0,
                 feed_forward_dimensions=1024):

        super(RecurrentMusicGenerator, self).__init__()

        self.pos_embedding = PositionalEncoding(d_model//2, max_len=seq_len)
        self.value_embedding = torch.nn.Embedding(n_tokens, d_model//2)

        self.transformer = RecurrentEncoderBuilder.from_kwargs(
            attention_type=attention_type,
            n_layers=n_layers,
            n_heads=n_heads,
            feed_forward_dimensions=feed_forward_dimensions,
            query_dimensions=d_query,
            value_dimensions=d_query,
            dropout=dropout,
            softmax_temp=softmax_temp,
            attention_dropout=attention_dropout,
        ).get()

        self.predictor = torch.nn.Linear(d_model, n_tokens)

    def forward(self, x, i=0, memory=None):
        x = x.view(x.shape[0])
        x = self.value_embedding(x)
        x = self.pos_embedding(x, i)

        y_hat, memory = self.transformer(x, memory)
        y_hat = self.predictor(y_hat)

        return y_hat, memory

def write_midi(words, path_midi, word2event):
    notes_all = []

    events = [word2event[words[i]] for i in range(len(words))]

    bar_cnt = 0
    cur_beat = 0

    midi_obj = miditoolkit.midi.parser.MidiFile()
    cur_pos = 0
    
    for i in range(len(events)-3):
        cur_event = events[i]
        # print(cur_event)
        name = cur_event.split('_')[0]
        attr = cur_event.split('_')
        if name == 'Bar':
            bar_cnt += 1
        elif name == 'Beat':
            cur_beat = int(attr[1])
            cur_pos = bar_cnt * BAR_RESOL + cur_beat * TICK_RESOL
        elif name == 'Chord':
            chord_text = attr[1] + '_' + attr[2]
            midi_obj.markers.append(Marker(text=chord_text, time=cur_pos))
        elif name == 'Tempo':
            midi_obj.tempo_changes.append(
                TempoChange(tempo=int(attr[1]), time=cur_pos))
        else:
            if 'Note_Pitch' in events[i] and \
            'Note_Velocity' in events[i+1] and \
            'Note_Duration' in events[i+2]:

                pitch = int(events[i].split('_')[-1])
                duration = int(events[i+2].split('_')[-1])

                if int(duration) == 0:
                    duration = 60

                end = cur_pos + duration 
                velocity = int(events[i+1].split('_')[-1])
                notes_all.append(
                    Note(pitch=pitch, start=cur_pos, end=end, velocity=velocity))
                
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = notes_all
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_midi)        

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

def sample_tokens(y_hat, num_samples=1):
    # Sample from filtered categorical distribution
    probs = torch.softmax(y_hat, dim=1)
    random_idx = torch.multinomial(probs, num_samples)
    return random_idx

def generate(model, prime, n, k=0, p=0, temperature=1.0):
    # Process prime sequence
    memory = None
    y_hat = []
    x_hat = []

    prime_len = prime.shape[1]
    for i in range(prime_len):
        x_hat.append(prime[:,i])
        y_i, memory = model(x_hat[-1], i=i, memory=memory)
        y_hat.append(y_i)

    # Generate new tokens
    for i in range(prime_len, prime_len + n):
        y_i = y_hat[-1]/temperature

        if k > 0:
            y_i = filter_top_k(y_i, k)
        if p > 0 and p < 1.0:
            y_i = filter_top_p(y_i, p)

        x_hat.append(sample_tokens(y_i))
        y_i, memory = model(x_hat[-1], i=i, memory=memory)
        y_hat.append(y_i)

    return [int(token) for token in x_hat]

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='generate.py')
    parser.add_argument('--model', type=str, required=True, help="Path to load model from.")
    parser.add_argument('--dict', type=str, required=True, help="Path to the dictionary.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--k', type=int, default=0, help="Number k of elements to consider while sampling.")
    parser.add_argument('--p', type=float, default=1.0, help="Probability p to consider while sampling.")
    parser.add_argument('--t', type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of transformer layers.")
    parser.add_argument('--d_query', type=int, default=32, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    args = parser.parse_args()

    # Set up torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dictionary
    event2word, word2event = pickle.load(open(args.dict, 'rb'))


    # Compute vocab size
    vocab_size = len(event2word.keys())

    # Build linear transformer
    model = RecurrentMusicGenerator(n_tokens=vocab_size,
                                     d_query=args.d_query,
                                     d_model=args.d_query * args.n_heads,
                                     seq_len=args.seq_len,
                              attention_type="linear",
                                    n_layers=args.n_layers,
                                     n_heads=args.n_heads).to(device)

    # Load model
    model.load_state_dict(torch.load(args.model, map_location=device)["model_state"])
    model.eval()

    # Define prime sequence
    prime = [event2word["Bar_None"]]
    prime = torch.tensor(prime).unsqueeze(dim=0).to(device)

    # Generate continuation
    words = generate(model, prime, n=1000, k=args.k, p=args.p, temperature=args.t)
    print(words)
    
    write_midi(words, "experiments/vgmidi/generated_piece.mid", word2event)

