import torch
import numpy as np
import plotext as plt

from generate import *

END_TOKEN = 389

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    def __init__(self, language_model, classifiers, emotion, vocab_size, device, gen_len=512, temperature=1.0, k=0, c=1):
        self.Qsa = {} # stores Q values for s,a (as defined in the paper)
        self.Nsa = {} # stores #times edge s,a was visited
        self.Ps  = {} # stores language model policy
        self.Ns  = {}

        self.language_model = language_model
        self.classifiers = classifiers
        self.emotion = emotion
        self.device = device

        self.k = k
        self.c = c
        self.gen_len = gen_len
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

        N = np.array([self.Nsa[(s, token)] if (s, token) in self.Nsa else 0 for token in range(self.vocab_size)])
        print(N)
        N = N**(1./self.temperature)
        N = N/np.sum(N)

        self.diff_distros(self.Ps[s].cpu().numpy(), N)

        next_token = np.random.choice(len(N), p=N)
        return next_token
        # return np.argmax(N)

    # def choose(self, state):
    #     "Choose the best successor of node. (Choose a move in the game)"
    #     s = self._get_string_representation(state)
    #
    #     N = np.array([self.Qsa[(s, token)] if (s, token) in self.Qsa else float("-inf") for token in range(self.vocab_size)])
    #     print(N)
    #
    #     return np.argmax(N)

    def _get_next_state(self, state, token):
        return torch.cat((state, torch.tensor([[token]]).to(self.device)), dim=1)

    def _is_terminal(self, state):
        return state.shape[-1] >= self.gen_len or state[-1,-1] == END_TOKEN

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

            # self.Qsa[s] = torch.zeros(self.vocab_size).to(self.device)
            # self.Nsa[s] = torch.zeros(self.vocab_size).to(self.device)

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
        #print("\t expand:", state)
        y_i = self.language_model(state)[:,-1,:]

        status_notes, _, _ = get_piece_status(state[-1].tolist())
        y_i = filter_note_off(y_i, status_notes)

        if self.k > 0:
            y_i = filter_top_k(y_i, self.k)

        y_i = torch.softmax(y_i, dim=1)

        return y_i.squeeze()

    def _rollout(self, state, depth=128):
        "Returns the reward for a random simulation (to completion) of `node`"
        memory = None
        piece = torch.clone(state)

        # Process current state
        log_prob = torch.zeros(1).to(self.device)
        for i in range(piece.shape[1]):
            x_i = piece[:,i:i+1]

            if i > 0:
                log_prob += torch.log(torch.softmax(y_i, dim=1)[:,x_i.squeeze()])

            y_i, memory = self.recurent_language_model(x_i, i=i, memory=memory)

        i = piece.shape[1]
        while (i % depth != 0) and (not self._is_terminal(piece)):
            status_notes, _, _ = get_piece_status(piece[-1].tolist())
            y_i = filter_note_off(y_i, status_notes)

            # Sample new token
            if self.k > 0:
                y_i = filter_top_k(y_i, self.k)

            # sample new token
            x_i = sample_tokens(y_i)

            # Accumulate probability
            log_prob += torch.log(torch.softmax(y_i, dim=1)[:,x_i.squeeze()])

            # Concatenate to current state
            piece = torch.cat((piece, x_i), dim=1)

            y_i, memory = self.recurent_language_model(x_i, i=i, memory=memory)
            i += 1

        return piece, log_prob

    def _reward(self, state):
        "Returns the reward for a random simulation (to completion) of `node`"
        # roll_state, roll_log_prob = self._rollout(state)
        print("continuation", state)

        # Emotion score
        clf_scores = torch.ones(1).to(self.device)
        for clf in self.classifiers:
            y_hat = clf(state)

            if y_hat.shape[-1] == 1:
                clf_scores *= torch.sigmoid(y_hat).squeeze()
            else:
                clf_scores *= torch.softmax(y_hat, dim=1)[:,self.emotion].squeeze()

        min_score = 0.0
        max_score = 1.0

        reward_fn = lambda x,a,b,c,d: (x - a) * (d - c) / (b - a) + c
        reward = reward_fn(clf_scores, min_score, max_score, -1.0, 1.0)

        print("reward", reward)
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
