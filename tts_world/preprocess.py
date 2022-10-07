"""Preprocess script for JSUT dataset."""

import argparse
import concurrent
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import ttslearn
from nnmnkwii.frontend import merlin
from nnmnkwii.io import hts
from tqdm import tqdm

_JSUT_BASIC5000_LABEL_DIR = Path.home() / "projects/jsut-label/labels/basic5000/"
_JSUT_BASIC5000_WAV_DIR = Path.home() / "datasets/jsut_ver1.1/basic5000/wav/"
_GENERATED_DIR = Path.cwd() / "generated"


def _duration(labels_path: Path) -> np.array:
    labels = hts.load(labels_path)
    dur = merlin.duration_features(labels)
    return dur


def _linguistic(labels_path: Path) -> np.array:
    binary_dict, numeric_dict = hts.load_question_set(ttslearn.util.example_qst_file())
    labels = hts.load(labels_path)
    lng = merlin.linguistic_features(labels, binary_dict, numeric_dict)
    return lng


def _acoustic(wav_path: Path) -> np.array:
    wav, sr = librosa.load(str(wav_path))
    # TODO: Use world_spss_params
    f0, sp, ap = pw.wav2world(wav.astype("double"), sr)
    aco = np.concatenate([np.expand_dims(f0, axis=-1), sp, ap], axis=-1)
    assert aco.shape[-1] == 1 + 513 + 513
    return aco


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    subparsers = parser.add_subparsers()

    duration_parser = subparsers.add_parser("duration")
    duration_parser.set_defaults(process=_duration)
    duration_parser.set_defaults(input_dir=_JSUT_BASIC5000_LABEL_DIR)
    duration_parser.set_defaults(output_dir=_GENERATED_DIR / "duration")

    linguistic_parser = subparsers.add_parser("linguistic")
    linguistic_parser.set_defaults(process=_linguistic)
    linguistic_parser.set_defaults(input_dir=_JSUT_BASIC5000_LABEL_DIR)
    linguistic_parser.set_defaults(output_dir=_GENERATED_DIR / "linguistic")

    acoustic_parser = subparsers.add_parser("acoustic")
    acoustic_parser.set_defaults(process=_acoustic)
    acoustic_parser.set_defaults(input_dir=_JSUT_BASIC5000_WAV_DIR)
    acoustic_parser.set_defaults(output_dir=_GENERATED_DIR / "acoustic")

    args = parser.parse_args()
    return args


def main() -> None:
    """The main function."""

    args = _get_args()
    input_paths = sorted(args.input_dir.glob("*"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as exec:
        results = list(tqdm(exec.map(args.process, input_paths)))

    for input_path, result in zip(input_paths, results):
        output_path = (args.output_dir / input_path.name).with_suffix(".npy")
        np.save(output_path, result)


if __name__ == "__main__":
    main()
