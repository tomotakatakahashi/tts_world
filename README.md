# TTS with WORLD

This repository is a Japanese Text-to-Speech (TTS) implementation.

[wav file for demo](./output.wav)

This is mostly a re-implementation of Chapters 5 and 6 of the book *"Pythonで学ぶ音声合成" (Speech Synthesis with Python)*. This project introduces the following modifications:

* Implements using **TensorFlow** instead of PyTorch.
* Bypasses the various acoustic features described in Chapter 5 to reduce the reliance on domain knowledge of speech processing. Instead, it utilizes `pyworld.wav2world` and `Conv1DTranspose`.

## Development Setup on Apple Silicon Mac
- `brew install python@3.9`
- `python3.9 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -U pip`
- `pip install tensorflow-macos black isort pytest pylint librosa notebook`
- Install [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) by following "Building from Source"
- `pip install -e .`
