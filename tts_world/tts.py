"""Inference Module. Generate speech from text and models."""

import argparse
from pathlib import Path

import numpy as np
import pyopenjtalk
import pyworld as pw
import soundfile as sf
import ttslearn
from nnmnkwii.frontend import merlin
from nnmnkwii.io import hts

_GENERATED_DIR = Path.cwd() / "generated"
_FloatType = np.float32

SAMPLING_RATE = 22050  # TODO: Integrate


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("acoustic_model", type=Path)
    parser.add_argument("duration_model", type=Path)
    parser.add_argument("text", type=str)
    parser.add_argument(
        "--generated-linguistic",
        type=Path,
        default=Path(__file__).resolve().parent / "../generated/linguistic",
    )
    parser.add_argument(
        "--generated-linguistic-frame",
        type=Path,
        default=Path(__file__).resolve().parent / "../generated/linguistic_frame",
    )
    parser.add_argument(
        "--generated-acoustic",
        type=Path,
        default=Path(__file__).resolve().parent / "../generated/acoustic",
    )
    parser.add_argument("--output-path", type=Path, default=Path.cwd() / "output.wav")
    args = parser.parse_args()
    return args


def _normalize(feature: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = np.divide(
        feature - np.expand_dims(mean, axis=0),
        np.expand_dims(std, axis=0),
        out=np.zeros_like(feature),
        where=(std != 0),
    )
    return normalized


def _unnormalize(
    normalized: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return mean + std * normalized


def _get_labels(text: str) -> hts.HTSLabelFile:
    contexts = pyopenjtalk.extract_fullcontext(text)
    labels = hts.HTSLabelFile.create_from_contexts(contexts)
    return labels


# TODO: Integrate with preprocess.py
def _get_linguistic_features(
    labels: hts.HTSLabelFile, add_frame_features=False, subphone_features=None
) -> np.ndarray:
    binary_dict, numeric_dict = hts.load_question_set(ttslearn.util.example_qst_file())
    lng = merlin.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=add_frame_features,
        subphone_features=subphone_features,
    )
    return lng


def main() -> None:
    """The main function."""

    args = _get_args()

    duration_model = tf.keras.models.load_model(args.duration_model)
    acoustic_model = tf.keras.models.load_model(args.acoustic_model)

    linguistic_mean = np.load(args.generated_linguistic / "mean.npy")
    linguistic_std = np.load(args.generated_linguistic / "std.npy")

    linguistic_frame_mean = np.load(args.generated_linguistic_frame / "mean.npy")
    linguistic_frame_std = np.load(args.generated_linguistic_frame / "std.npy")

    acoustic_mean = np.load(args.generated_acoustic / "mean.npy")
    acoustic_std = np.load(args.generated_acoustic / "std.npy")

    labels = _get_labels(args.text)

    linguistic_features = _get_linguistic_features(labels)
    durations_normalized = duration_model.predict(
        _normalize(linguistic_features, linguistic_mean, linguistic_std)
    )
    durations_predicted = _unnormalize(
        durations_normalized, linguistic_mean, linguistic_std
    )

    labels.set_durations(durations_predicted)
    linguistic_frame_features = _get_linguistic_features(
        labels, add_frame_features=True, subphone_features="coarse_coding"
    )

    acoustic_normalized = acoustic_model.predict(
        _normalize(
            linguistic_frame_features, linguistic_frame_mean, linguistic_frame_std
        )
    )
    acoustic_normalized = np.concatenat(
        acoustic_normalized, axis=-1
    )  # Concat (f0, sp, ap)
    acoustic_predicted = _unnormalize(acoustic_normalized, acoustic_mean, acoustic_std)

    f0 = acoustic_predicted[:, 0]
    sp = acoustic_predicted[:, 1:514]
    ap = acoustic_predicted[:, 514:]

    audio = pw.synthesize(f0, sp, ap, SAMPLING_RATE)
    sf.write(args.output_path, audio, SAMPLING_RATE, subtype="PCM_24")


if __name__ == "__main__":
    main()
