"""Preprocess script for JSUT dataset."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin

_JSUT_BASIC5000_LABEL_DIR = Path.home() / "projects/jsut-label/labels/basic5000/"
_GENERATED_DIR = Path.cwd() / "generated"


def _duration(label_path: Path) -> np.array:
    label = hts.load(label_path)
    dur = merlin.duration_features(label)
    return dur


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    subparsers = parser.add_subparsers()

    duration_parser = subparsers.add_parser("duration")
    duration_parser.set_defaults(process=_duration)
    duration_parser.set_defaults(input_dir=_JSUT_BASIC5000_LABEL_DIR)
    duration_parser.set_defaults(output_dir=_GENERATED_DIR / "duration")

    args = parser.parse_args()
    return args


def main() -> None:
    """The main function."""

    args = _get_args()
    input_paths = sorted(args.input_dir.glob("*"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [args.process(input_path) for input_path in input_paths]

    for input_path, result in zip(input_paths, results):
        output_path = (args.output_dir / input_path.name).with_suffix(".npy")
        np.save(output_path, result)


if __name__ == "__main__":
    main()


"""
    wav_path = Path.home() / "datasets/jsut_ver1.1/basic5000/wav/BASIC5000_0001.wav"
    wav, sr = librosa.load(str(wav_path))
    f0, sp, ap = pw.wav2world(wav.astype("double"), sr)
"""
