# speech-to-text
This project implements an Automatic Speech Recognition (ASR) system in Python using TensorFlow/Keras. It trains a DeepSpeech2-like model on the LJSpeech dataset to convert speech audio into text transcriptions.

# Features

Preprocessing: Audio normalization, spectrogram generation (STFT).

Character-level encoding using StringLookup.

Custom CTC Loss function for sequence-to-sequence training.

DeepSpeech2-inspired architecture:

Convolutional layers

Bidirectional GRUs

Fully-connected softmax output

Model evaluation using Word Error Rate (WER).

Visualization of spectrograms & audio signals.

# Tools & Libraries

Python

TensorFlow / Keras

NumPy & Pandas

Matplotlib

jiwer (for WER calculation)

IPython display (for audio playback)

# Files

train.py → Data preprocessing + model training.

model.py → DeepSpeech2 model definition.

utils.py → Helper functions (encoding, decoding, CTC loss).

README.md → Project documentation.
