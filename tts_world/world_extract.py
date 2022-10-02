"""Extract features using pyworld."""

import argparse
from pathlib import Path

import librosa
import pyworld as pw


def _get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def world_extract():
    wav_path = Path.home() / "datasets/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
    wav, sr = librosa.load(wav_path)
    f0, sp, ap = pw.wav2world(wav, sr)

    print(f0.shape)
    print("TODO")
