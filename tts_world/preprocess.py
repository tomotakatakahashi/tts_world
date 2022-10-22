"""Preprocess script for JSUT dataset."""

import argparse
import concurrent.futures
from pathlib import Path
from typing import Callable, List, Optional

import librosa
import numpy as np
import pyworld as pw
import ttslearn
from nnmnkwii.frontend import merlin
from nnmnkwii.io import hts
from nnmnkwii.preprocessing.f0 import interp1d
from tqdm import tqdm

_GENERATED_DIR = Path.cwd() / "generated"
_FloatType = np.float32

_TRAIN_RANGE = (0, 4000)
_VAL_RANGE = (4000, 4500)
_TEST_RANGE = (4500, 5000)


def _duration(labels_path: Path, _: None) -> np.ndarray:
    labels = hts.load(labels_path)
    dur = merlin.duration_features(labels).astype(_FloatType)
    return dur


def _linguistic_impl(
    labels_path: Path,
    add_frame_features: bool = False,
    subphone_features: Optional[str] = None,
) -> np.ndarray:
    binary_dict, numeric_dict = hts.load_question_set(ttslearn.util.example_qst_file())
    labels = hts.load(labels_path)
    lng = merlin.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=add_frame_features,
        subphone_features=subphone_features,
    )
    return lng.astype(_FloatType)


def _linguistic(labels_path: Path, _: None) -> np.ndarray:
    return _linguistic_impl(labels_path)


def _linguistic_frame(labels_path: Path, _: None) -> np.ndarray:
    return _linguistic_impl(
        labels_path, add_frame_features=True, subphone_features="coarse_coding"
    )


def _acoustic(wav_path: Path, lng_path: Path) -> np.ndarray:
    wav, sr = librosa.load(str(wav_path))
    # TODO: Use world_spss_params
    f0, sp, ap = pw.wav2world(wav.astype("double"), sr)  # pylint: disable=no-member
    voice_flag = (f0 != 0).astype(_FloatType)
    lf0 = np.log(interp1d(f0, kind="linear"))
    sp = np.log(sp)
    aco = np.concatenate(
        [np.expand_dims(voice_flag, axis=-1), np.expand_dims(lf0, axis=-1), sp, ap],
        axis=-1,
    )
    assert aco.shape[-1] == 1 + 1 + 513 + 513

    # Truncate - is this correct?
    linguistic_frame = np.load(lng_path)
    assert len(linguistic_frame) <= len(aco)
    aco = aco[: len(linguistic_frame)]

    return aco.astype(_FloatType)


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--extra-dir", type=Path, default=None)
    parser.add_argument("--slice", type=int, default=None)
    subparsers = parser.add_subparsers()

    duration_parser = subparsers.add_parser("duration")
    duration_parser.set_defaults(process=_duration)

    linguistic_parser = subparsers.add_parser("linguistic")
    linguistic_parser.set_defaults(process=_linguistic)

    linguistic_frame_parser = subparsers.add_parser("linguistic_frame")
    linguistic_frame_parser.set_defaults(process=_linguistic_frame)

    acoustic_parser = subparsers.add_parser("acoustic")
    acoustic_parser.set_defaults(process=_acoustic)

    args = parser.parse_args()
    return args


def statistics_axis(arrays: List[np.ndarray], func: Callable) -> np.ndarray:
    """
    Memory-efficient impl of np.mean or np.std.
    cf.
    arrays_concat = np.concatenate(arrays, axis=0)
    mean = np.mean(arrays_concat, axis=0)
    std = np.std(arrays_concat, axis=0)
    """
    assert func in (np.mean, np.std)
    assert isinstance(arrays, list)
    assert len(arrays) > 0

    result = np.zeros_like(arrays[0][0], dtype=_FloatType)
    for i in range(result.shape[0]):
        arrays_concat = np.concatenate([array[:, i] for array in arrays], axis=0)
        result[i] = func(arrays_concat, axis=0, dtype=_FloatType)
    return result


def main() -> None:
    """The main function."""

    args = _get_args()
    input_paths = sorted(args.input_dir.glob("*"))
    extra_paths = (
        sum(
            (
                sorted((args.extra_dir / kind).glob("*"))
                for kind in ["train", "val", "test"]
            ),
            [],
        )
        if args.extra_dir is not None
        else [None] * len(input_paths)
    )
    assert len(input_paths) == len(extra_paths)
    if args.slice is not None:
        input_paths = input_paths[: args.slice]
        extra_paths = extra_paths[: args.slice]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(args.process, input_paths, extra_paths),
                total=len(input_paths),
            )
        )

    mean = statistics_axis(results[_TRAIN_RANGE[0] : _TRAIN_RANGE[1]], np.mean)
    std = statistics_axis(results[_TRAIN_RANGE[0] : _TRAIN_RANGE[1]], np.std)
    np.save(args.output_dir / "mean.npy", mean.astype(_FloatType))
    np.save(args.output_dir / "std.npy", std.astype(_FloatType))

    for i, (input_path, result) in tqdm(
        enumerate(zip(input_paths, results)), total=len(input_paths)
    ):
        result_normalized = np.divide(
            result - np.expand_dims(mean, axis=0),
            np.expand_dims(std, axis=0),
            out=np.zeros_like(result, dtype=_FloatType),
            where=(std != 0),
        )
        if _TRAIN_RANGE[0] <= i < _TRAIN_RANGE[1]:
            output_dir = args.output_dir / "train"
        elif _VAL_RANGE[0] <= i < _VAL_RANGE[1]:
            output_dir = args.output_dir / "val"
        elif _TEST_RANGE[0] <= i < _TEST_RANGE[1]:
            output_dir = args.output_dir / "test"
        else:
            raise RuntimeError
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (output_dir / input_path.name).with_suffix(".npy")
        np.save(output_path, result_normalized.astype(_FloatType))


if __name__ == "__main__":
    main()
