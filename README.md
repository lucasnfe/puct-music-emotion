# Controlling Perceived Emotion in Symbolic Music Generation with Monte Carlo Tree Search

This repository contains the source code to reproduce the results of the [AIIDE'22](https://sites.google.com/view/aiide-2022/) paper [Controlling Perceived Emotion in Symbolic Music Generation with Monte Carlo Tree Search](https://arxiv.org/abs/2208.05162). This paper presents a new approach for controlling emotion in symbolic music generation with Monte Carlo Tree Search. We use Monte Carlo Tree Search as a decoding mechanism to steer the probability distribution learned by a language model towards a given emotion. At every step of the decoding process, we use Predictor Upper Confidence for Trees (PUCT) to search for sequences that maximize the average values of emotion and quality as given by an emotion classifier and a discriminator, respectively. We use a language model as PUCT's policy and a combination of the emotion classifier and the discriminator as its value function. To decode the next token in a piece of music, we sample from the distribution of node visits created during the search. We evaluate the quality of the generated samples with respect to human-composed pieces using a set of objective metrics computed directly from the generated samples. We also perform a user study to evaluate how human subjects perceive the generated samples' quality and emotion. We compare PUCT against Stochastic Bi-Objective Beam Search (SBBS) and Conditional Sampling (CS). Results suggest that PUCT outperforms SBBS and CS in almost all metrics of music quality and emotion.

## Examples of Generated Pieces

In this paper, we discretize the Circumplex (valence-arousal) model of emotion into four quadrants, which yelds four emotion classes: high valence and arousal **(E1)**, low valence and high arousal **(E2)**, low valence and arousal **(E3)**, and high valence and low arousal **(E4)**.

| **Emotion E1**  | **Emotion E2** | **Emotion E3**  | **Emotion E4** |
| ------------- | ------------- | ------------- | ------------- |
| [Piece e1_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e1_fake_mcts_7.mp3)  | [Piece e2_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e2_fake_mcts_3.mp3)  | [Piece e3_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e3_fake_mcts_7.mp3)  | [Piece e4_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e4_fake_mcts_1.mp3)  |
| [Piece e2_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e1_fake_mcts_8.mp3)  | [Piece e2_2](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e2_fake_mcts_4.mp3)  | [Piece e3_2](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e3_fake_mcts_1.mp3)  | [Piece e4_2](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e4_fake_mcts_2.mp3) | 
| [Piece e3_3](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e1_fake_mcts_10.mp3)  | [Piece e2_3](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e2_fake_mcts_7.mp3)  | [Piece e3_3](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e3_fake_mcts_17.mp3)  | [Piece e4_3](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e4_fake_mcts_3.mp3) | 

## Reproducing Results

Our PUCT approach uses three neural models: a music language model (our "policy" network), a music emotion classifier and a music discriminator (our "value" networks). The easiest way to reproduce the results is to download the trained models from the `trained` folder and run the following command:

``
 python3 -u generate_mcts.py --lm trained/language_model.pth --clf trained/emotion_classifier.pth --disc trained/discriminator_classifier.pth
                             --emotion 1 --seq_len 1024 --save_to e1_mcts_1.mid --p 0.9 --n_bars 16
``

This command will generate a piece with emotion E1 (--emotion 1) using PUCT and the trained models. This piece will be saved as a mid file names 'e1_mcts_1.mid'. To generate pieces with different emotions, change the value of the `--emotion` argument to 2, 3, or 4.

## Training Models from Scratch

#### Download VGMIDI data set

#### 1. Language Model

#### 2. Train Emotion Classifier

#### 3. Train Emotion Classifier

## Citing this Work

If you use our PUCT method in your research, please cite:

```
@inproceedings{ferreira2022puct,
  title={Controlling Perceived Emotion in Symbolic Music Generation with Monte Carlo Tree Search},
  author={N. Ferreira, Lucas and Mou, Lili and Whitehead, Jim and Lelis, Levi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
  year={2022}
}
```
