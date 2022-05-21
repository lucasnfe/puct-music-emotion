import torch
import numpy as np
import plotext as plt

from encoder import *
from generate import *

END_TOKEN = Event(event_type='control', value=2).to_int()
BAR_TOKEN = Event(event_type='control', value=1).to_int()

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    def __init__(self, language_model, emotion_classifier, discriminator, emotion, vocab_size, device, n_bars, seq_len=1024, p=0, c=1):
        self.Qsa = {} # stores Q values for s,a (as defined in the paper)
        self.Nsa = {} # stores #times edge s,a was visited
        self.Ps  = {} # stores language model policy
        self.Ns  = {}

        self.language_model = language_model
        self.emotion_classifier = emotion_classifier
        self.discriminator = discriminator
        self.emotion = emotion
        self.device = device

        self.p = p
        self.c = c
        self.seq_len = seq_len
        self.n_bars = n_bars
        self.vocab_size = vocab_size

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

    def choose(self, state, temperature=1.0):
        "Choose the best successor of node. (Choose a move in the game)"
        s = self._get_string_representation(state)

        N = torch.tensor([self.Nsa[(s, token)] if (s, token) in self.Nsa else 0 for token in range(self.vocab_size)], device=self.device, dtype=torch.float)
        M = torch.tensor([float(self.Qsa[(s, token)]) if (s, token) in self.Qsa else float('-inf') for token in range(self.vocab_size)], device=self.device)
        print(N)
        print(M)
        N = N**(1./temperature)
        
        #self.diff_distros(self.Ps[s], N)
        
        #next_token = torch.multinomial(N, num_samples=1)
        next_token = torch.argmax(N)
        #next_token = torch.argmax(M)
        
        return int(next_token)

    def _get_next_state(self, state, token):
        return torch.cat((state, torch.tensor([token], device=self.device)), dim=0)

    def _is_terminal(self, state):
        return torch.sum(state == BAR_TOKEN) >= self.n_bars or len(state) >= self.seq_len or state[-1] == END_TOKEN

    def _get_string_representation(self, state):
        return " ".join([str(int(token)) for token in state])

    def step(self, state):
        s = self._get_string_representation(state)

        "Make the tree one layer better. (Train for one iteration.)"
        if self._is_terminal(state):
            v = self._reward(state)
            return v

        if s not in self.Ps:
            # leaf node
            self.Ps[s] = self._expand(state)
            self.Ns[s] = 0

            v = self._reward(state)
            return v

        # Select next token
        token = self._select(s)

        # Recursevily call step until a leaf node is found
        next_state = self._get_next_state(state, token)

        #print("\t selected:", token)
        v = self.step(next_state)

        if (s, token) in self.Qsa:
            self.Qsa[(s, token)] = (self.Nsa[(s, token)] * self.Qsa[(s, token)] + v) / (self.Nsa[(s, token)] + 1)
            self.Nsa[(s, token)] += 1
        else:
            self.Qsa[(s, token)] = v
            self.Nsa[(s, token)] = 1

        self.Ns[s] += 1

        return v

    def _expand(self, state):
        with torch.no_grad():
            print("\t expand:", state)
            y_i = self.language_model(state.unsqueeze(0))[:,-1,:]
        
            # Filter out end token
            y_i = filter_index(y_i, END_TOKEN)
            
            # Filter top_k
            if self.p > 0:
                y_i = filter_top_p(y_i, p=self.p)
       
            return torch.softmax(y_i, dim=1).squeeze()

    def _rollout(self, state, depth=1):
        if int(state[-1]) == BAR_TOKEN:
            return state.unsqueeze(0)
        
        with torch.no_grad():
            n_bars = 0
            
            roll_state = torch.clone(state).unsqueeze(0)
            
            while n_bars < depth and not self._is_terminal(roll_state.squeeze()):
                y_i = self.language_model(roll_state)[:,-1,:]

                # Filter out end token
                y_i = filter_index(y_i, END_TOKEN)
                
                # Filter top_k
                if self.p > 0:
                    y_i = filter_top_p(y_i, p=self.p)

                token = sample_tokens(y_i)
                if int(token) == BAR_TOKEN:
                    n_bars += 1

                # Concatenate to current state
                roll_state = torch.cat((roll_state, token), dim=1)
                
            return roll_state

    def _reward(self, state):
        "Returns the reward for a random simulation (to completion) of `node`"
        roll_state = self._rollout(state, depth=1)
        print("continuation", roll_state)
        
        # Discriminator score
        y_hat = self.discriminator(roll_state)
        discriminator_score = torch.sigmoid(y_hat).squeeze()

        # Emotion score
        y_hat = torch.softmax(self.emotion_classifier(roll_state), dim=1).squeeze()
        
        emotion_hat = int(torch.argmax(y_hat))
        emotion_score = y_hat[self.emotion]

        if emotion_hat == self.emotion:
            #reward = emotion_score * discriminator_score
            reward = discriminator_score
        else:
            reward = (emotion_score - 1.0)# * (1.0 - discriminator_score)

        print("reward", emotion_hat, discriminator_score, reward)
        #print("reward", emotion_hat, emotion_score, reward)
        return reward

    def _select(self, s, eps=1e-8):
        cur_best = -float('inf')
        best_token = -1
        
        top_k_tokens = torch.where(self.Ps[s] > 0)[0].cpu().numpy()

        for token in top_k_tokens:
            if (s, token) in self.Qsa:
                u = self.Qsa[(s, token)] + self.c * self.Ps[s][token] * np.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, token)])
            else:
                u = self.c * self.Ps[s][token] * np.sqrt(self.Ns[s] + eps)

            if u > cur_best:
                cur_best = u
                best_token = token

        return int(best_token)
