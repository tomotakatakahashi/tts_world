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

_GENERATED_DIR = Path.cwd() / "generated"


def _duration(labels_path: Path) -> np.array:
    labels = hts.load(labels_path)
    dur = merlin.duration_features(labels).astype("float")
    return dur


def _linguistic_impl(
    labels_path: Path, add_frame_features=False, subphone_features=None
) -> np.array:
    binary_dict, numeric_dict = hts.load_question_set(ttslearn.util.example_qst_file())
    labels = hts.load(labels_path)
    lng = merlin.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=add_frame_features,
        subphone_features=subphone_features,
    )
    return lng


def _linguistic(labels_path: Path) -> np.array:
    return _linguistic_impl(labels_path)


def _linguistic_frame(labels_path: Path) -> np.array:
    return _linguistic_impl(
        labels_path, add_frame_features=True, subphone_features="coarse_coding"
    )


def _acoustic(wav_path: Path) -> np.array:
    wav, sr = librosa.load(str(wav_path))
    # TODO: Use world_spss_params
    # TODO: time length is the same as linguistic_frame?
    f0, sp, ap = pw.wav2world(wav.astype("double"), sr)
    aco = np.concatenate([np.expand_dims(f0, axis=-1), sp, ap], axis=-1)
    assert aco.shape[-1] == 1 + 513 + 513
    return aco


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
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


def statistics_axis(arrays, func):
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

    result = np.zeros_like(arrays[0][0], dtype="double")
    for i in range(len(result)):
        arrays_concat = np.concatenate([array[:, i] for array in arrays], axis=0)
        result[i] = func(arrays_concat, axis=0)
    return result


def main() -> None:
    """The main function."""

    args = _get_args()
    input_paths = sorted(args.input_dir.glob("*"))
    if args.slice is not None:
        input_paths = input_paths[: args.slice]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor() as exec:
        results = list(
            tqdm(exec.map(args.process, input_paths), total=len(input_paths))
        )

    # TODO: Stop using test data in train data
    mean = statistics_axis(results, np.mean)
    std = statistics_axis(results, np.std)
    np.save(args.output_dir / "mean.npy", mean.astype(np.float32))
    np.save(args.output_dir / "std.npy", std.astype(np.float32))

    for input_path, result in tqdm(zip(input_paths, results), total=len(input_paths)):
        result_normalized = np.divide(
            result - np.expand_dims(mean, axis=0),
            np.expand_dims(std, axis=0),
            out=np.zeros_like(result),
            where=(std != 0),
        )
        output_path = (args.output_dir / input_path.name).with_suffix(".npy")
        np.save(output_path, result_normalized.astype(np.float32))


if __name__ == "__main__":
    main()
