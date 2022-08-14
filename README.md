# Controlling Perceived Emotion in Symbolic Music Generation with Monte Carlo Tree Search

This repository contains the source code to reproduce the results of the [AIIDE'22](https://sites.google.com/view/aiide-2022/) paper [Controlling Perceived Emotion in Symbolic Music Generation with Monte Carlo Tree Search](https://arxiv.org/abs/2208.05162). This paper presents a new approach for controlling emotion in symbolic music generation with Monte Carlo Tree Search. We use Monte Carlo Tree Search as a decoding mechanism to steer the probability distribution learned by a language model towards a given emotion. At every step of the decoding process, we use Predictor Upper Confidence for Trees (PUCT) to search for sequences that maximize the average values of emotion and quality as given by an emotion classifier and a discriminator, respectively. We use a language model as PUCT's policy and a combination of the emotion classifier and the discriminator as its value function. To decode the next token in a piece of music, we sample from the distribution of node visits created during the search. We evaluate the quality of the generated samples with respect to human-composed pieces using a set of objective metrics computed directly from the generated samples. We also perform a user study to evaluate how human subjects perceive the generated samples' quality and emotion. We compare PUCT against Stochastic Bi-Objective Beam Search (SBBS) and Conditional Sampling (CS). Results suggest that PUCT outperforms SBBS and CS in almost all metrics of music quality and emotion.

## Examples of Generated Pieces

In this paper, we discretize the Circumplex (valence-arousal) model of emotion into four quadrants, which yelds four emotion classes: high valence and arousal **(E1)**, low valence and high arousal **(E2)**, low valence and arousal **(E3)**, and high valence and low arousal **(E4)**.

| **Emotion E1**  | **Emotion E2** | **Emotion E3**  | **Emotion E4** |
| ------------- | ------------- | ------------- | ------------- |
| [Piece e1_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e1_fake_mcts_7.mp3)  | [Piece e2_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e2_fake_mcts_3.mp3)  | [Piece e3_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e3_fake_mcts_7.mp3)  | [Piece e4_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e4_fake_mcts_1.mp3)  |
| [Piece e2_1](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e1_fake_mcts_8.mp3)  | [Piece e2_2](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e2_fake_mcts_4.mp3)  | [Piece e3_2](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e3_fake_mcts_1.mp3)  | [Piece e4_2](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e4_fake_mcts_2.mp3) | 
| [Piece e3_3](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e1_fake_mcts_10.mp3)  | [Piece e2_3](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e2_fake_mcts_7.mp3)  | [Piece e3_3](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e3_fake_mcts_17.mp3)  | [Piece e4_3](https://raw.githubusercontent.com/lucasnfe/aiide22/main/experiments/user_study/static/audio/mcts/e4_fake_mcts_3.mp3) | 

## Dependencies

```
pip install requirements.txt
```

## Reproducing Results

Our PUCT approach uses three neural models: a music language model (our "policy" network), a music emotion classifier and a music discriminator (our "value" networks). The easiest way to reproduce the results is to [download the trained models](https://drive.google.com/drive/folders/1bgx-r2gFi6yFTFGTOZbnrxUVvue-Dold?usp=sharing) and run the `generate_mcts.py` script from the `workspace` folder:

```
 python3 generate_mcts.py --lm language_model_epoch_6.pth \ 
                          --clf emotion_classifier_epoch_83.pth 
                          --disc discriminator_epoch_3.pth \
                          --emotion 1 --seq_len 1024 --n_bars 16 \
                          --p 0.9 --c 1 --roll_steps 50 \
                          --save_to e1_mcts_1.mid
```

The `generate_mcts.py` script will generate a piece with emotion E1 (--emotion 1) using PUCT and the trained models. This piece will be saved as a mid file names 'e1_mcts_1.mid'. To generate pieces with different emotions, change the value of the `--emotion` argument to 2, 3, or 4.

## Training the Models

To retrain the models from the data you will need to download and pre-process the VGMIDI dataset.

### VGMIDI Dataset 

**1. Download the VGMIDI dataset**

```
$ cd dataset
$ wget https://github.com/lucasnfe/puct-music-emotion/releases/download/aiide22/vgmidi_clean.zip
$ unzip vgmidi_clean.zip
```

To simplify our music language modeling task, we trained the LM using only the VGMIDI pieces with 4/4 time signature. This subset 
has 2,520 pieces, of which we used 2,142 (85%) for training and 378 (15%) for testing. We trained the emotion classifier with the 200 labeled pieces of the VGMIDI data set. We used 140 (70%) pieces for training and 60 (30%)for testing.  The discriminator was trained with 400 pieces, the 200 labeled pieces (real) of the VGMIDI data set, and other 200 (fake) pieces generated via Top-p sampling with p = 0.9.

**2. Data Pre-processing**

The pre-processing step consists of augmenting the data, encoding it with REMI and compiling the encoded pieces as a numpy array.

**2.1 Data Augmentation**

All unlabelled pieces were augmented by (a) transposing to every key, (b) increasing and decreasing the tempo by 10%, and (c) increasing and decreasing the velocity of all notes by 10%, as Oore et al. (2017) described.

```
$ python3 augment.py --path_indir vgmidi_clean/unlabelled --path_outdir vgmidi_augmented/unlabelled
```

Only the unlabelled pieces are augmented. We don't augmented the labelled pieces because augmented versions might not have the same emotion of the original ones. To keep all pieces in the same directory, copy the labelled and generated peices to the `augmented` directory.


```
$ cp -r vgmidi_clean/labelled vgmidi_augmented/
$ cp -r vgmidi_clean/fake_top_p vgmidi_augmented/
```

**2.2 REMI Encoding**

We encoded all pieces using REMI (Huang and Yang 2020).

```
$ python3 encoder.py --path_indir vgmidi_augmented --path_outdir vgmidi_encoded
```

**2.3 Compile all pieces in a numpy array.**

```
$ python3 compile.py --path_indir encoded --path_outdir encoded
```

#### 1. Language Model

#### 2. Train Emotion Classifier

#### 3. Train Discriminator

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
