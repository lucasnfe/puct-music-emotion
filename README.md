# Controlling Perceived Emotion in Symbolic Music Generation with Monte Carlo Tree Search

This repository contains the source code to reproduce the results of the [AIIDE'22](https://sites.google.com/view/aiide-2022/)
paper [Controlling Perceived Emotion in Symbolic Music Generation with Monte Carlo Tree Search](https://arxiv.org/abs/2208.05162). This paper presents a new approach 
for controlling emotion in symbolic music generation with Monte Carlo Tree Search. We use Monte Carlo Tree Search as a decoding 
mechanism to steer the probability distribution learned by a language model towards a given emotion. At every step of the 
decoding process, we use Predictor Upper Confidence for Trees (PUCT) to search for sequences that maximize the average values 
of emotion and quality as given by an emotion classifier and a discriminator, respectively. We use a language model as PUCT's 
policy and a combination of the emotion classifier and the discriminator as its value function. To decode the next token in 
a piece of music, we sample from the distribution of node visits created during the search.

## Examples of Generated Pieces

In this paper, we discretize the Circumplex (valence-arousal) model of emotion into four
quadrants, which yelds four emotion classes: high valence and arousal *(E1)*, low valence and high arousal *(E2)*, low valence and arousal *(E3)*, and high valence and low arousal *(E4)*.

### Emotion E1 
- [Piece e1_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e1_fake_mcts_7.mp3)
- [Piece e2_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e1_fake_mcts_8.mp3)
- [Piece e3_3](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e1_fake_mcts_10.mp3)

### Emotion E2
- [Piece e2_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e2_fake_mcts_3.mp3)
- [Piece e2_2](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e2_fake_mcts_4.mp3)
- [Piece e2_3](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e2_fake_mcts_7.mp3)

### Emotion E3
- [Piece e3_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e3_fake_mcts_7.mp3)
- [Piece e3_2](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e3_fake_mcts_1.mp3)
- [Piece e3_3](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e3_fake_mcts_17.mp3)

### Emotion E4
- [Piece e4_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e4_fake_mcts_1.mp3)
- [Piece e4_2](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e4_fake_mcts_2.mp3)
- [Piece e4_3](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e4_fake_mcts_3.mp3)
