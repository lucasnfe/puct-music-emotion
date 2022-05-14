import torch
import numpy as np
import plotext as plt

from encoder import *
from generate import *

END_TOKEN = Event(event_type='control', value=2).to_int()
BAR_TOKEN = Event(event_type='control', value=1).to_int()

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    def __init__(self, language_model, emotion_classifier, discriminator, emotion, vocab_size, device, n_bars, seq_len=1024, temperature=1.0, k=0, c=1):
        self.Qsa = {} # stores Q values for s,a (as defined in the paper)
        self.Nsa = {} # stores #times edge s,a was visited
        self.Ps  = {} # stores language model policy
        self.Ns  = {}

        self.language_model = language_model
        self.emotion_classifier = emotion_classifier
        self.discriminator = discriminator
        self.emotion = emotion
        self.device = device

        self.k = k
        self.c = c
        self.seq_len = seq_len
        self.n_bars = n_bars
        self.vocab_size = vocab_size
        self.temperature = temperature

    def diff_distros(self, old, new):
        tokens = [i for i in range(self.vocab_size)]

        plt.clf()
        plt.subplots(1, 2)

        plt.subplot(1, 1)
        plt.clc()
        plt.ylim(0.0,1.0)
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        plt.title("Old token distribution")
        plt.plot(tokens, np.array(old, dtype=np.float64), marker='dot')

        plt.subplot(1, 2)
        plt.clc()
        plt.ylim(0.0,1.0)
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        plt.title("New token distribution")
        plt.plot(tokens, np.array(new, dtype=np.float64), marker='dot')

        plt.show()

    def choose(self, state):
        "Choose the best successor of node. (Choose a move in the game)"
        s = self._get_string_representation(state)

        N = torch.zeros(1, self.vocab_size).to(self.device)
        for token in range(self.vocab_size):
            if (s, token) in self.Nsa:
                N[:,token] = self.Nsa[(s, token)]
       
        N = N/torch.sum(N)
        print(N)
        
        self.diff_distros(self.Ps[s].cpu().numpy(), N.squeeze(0).cpu().numpy())

        next_token = torch.multinomial(N, num_samples=1)

        return int(next_token)

    def _get_next_state(self, state, token):
        return torch.cat((state, torch.tensor([[token]]).to(self.device)), dim=1)

    def _is_terminal(self, state):
        return torch.sum(state == BAR_TOKEN) >= self.n_bars or state.shape[-1] >= self.seq_len or state[-1,-1] == END_TOKEN

    def _get_string_representation(self, state):
        return " ".join([str(int(token)) for token in state[-1]])

    def step(self, state):
        s = self._get_string_representation(state)

        "Make the tree one layer better. (Train for one iteration.)"
        if self._is_terminal(state):
            value = self._reward(state)
            return value

        if s not in self.Ps:
            # leaf node
            self.Ps[s] = self._expand(state)
            self.Ns[s] = 0

            value = self._reward(state)
            return value

        # Select next token
        token = self._select(s)

        # Recursevily call step until a leaf node is found
        next_state = self._get_next_state(state, token)

        #print("\t selected:", token)
        value = self.step(next_state)

        if (s, token) in self.Qsa:
            self.Qsa[(s, token)] = (self.Nsa[(s, token)] * self.Qsa[(s, token)] + value) / (self.Nsa[(s, token)] + 1)
            self.Nsa[(s, token)] += 1

        else:
            self.Qsa[(s, token)] = value
            self.Nsa[(s, token)] = 1

        self.Ns[s] += 1

        return value

    def _expand(self, state):
        print("\t expand:", state)
        y_i = self.language_model(state)[:,-1,:]
        
        # Filter out end token
        y_i[-1][END_TOKEN] = float('-inf')

        if self.k > 0:
            y_i = filter_top_k(y_i, self.k)

        y_i = torch.softmax(y_i, dim=1)

        return y_i.squeeze()

    def _rollout(self, state, depth=1):
        piece = torch.clone(state)

        n_bars = 0
        while (n_bars == 0 or n_bars % depth != 0) and not self._is_terminal(piece):
            y_i = self.language_model(piece)[:,-1,:]
            
            # Filter out end token
            y_i[-1][END_TOKEN] = float('-inf')
            
            # Sample new token
            if self.k > 0:
                y_i = filter_top_k(y_i, self.k)
            
            token = sample_tokens(y_i)

            if int(token) == BAR_TOKEN:
                n_bars += 1

            # Concatenate to current state
            piece = torch.cat((piece, token), dim=1)

        return piece

    def _reward(self, state):
        "Returns the reward for a random simulation (to completion) of `node`"
        roll_state = self._rollout(state, depth=1)
        print("continuation", roll_state)
        
        # Discriminator score
        y_hat = self.discriminator(roll_state)
        discriminator_score = torch.sigmoid(y_hat).squeeze()

        # Emotion score
        y_hat = self.emotion_classifier(roll_state)
        _, emotion_score = torch.max(y_hat.view(-1, 4).data, dim=1)
        #emotion_score = torch.softmax(y_hat, dim=1)[:,self.emotion].squeeze()

        reward = 0.0
        if int(emotion_score) == self.emotion:
            reward = 1.0 * discriminator_score
        else:
            reward = -1.0 * (1.0 - discriminator_score)
        
        #min_score = 0.0
        #max_score = 1.0

        #reward_fn = lambda x,a,b,c,d: (x - a) * (d - c) / (b - a) + c
        #reward = reward_fn(emotion_score * discriminator_score, min_score, max_score, -1.0, 1.0)
        #reward = emotion_score * discriminator_score

        print("reward", emotion_score, discriminator_score, reward)
        return reward

    def _select(self, s, eps=1e-8):
        cur_best = -float('inf')
        best_token = -1

        top_k_filtered_tokens = np.where(self.Ps[s].cpu().numpy() > 0)[0]

        for token in top_k_filtered_tokens:
            if (s, token) in self.Qsa:
                u = self.Qsa[(s, token)] + self.c * self.Ps[s][token] * np.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, token)])
            else:
                u = self.c * self.Ps[s][token] * np.sqrt(self.Ns[s] + eps)

            if u > cur_best:
                cur_best = u
                best_token = token

        return best_token

