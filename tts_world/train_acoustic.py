"""Train an acoustic model."""

from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError as MSE

GENERATED_DIR = Path.cwd() / "generated"

ACOUSTIC_DIR = GENERATED_DIR / "acoustic"
LINGUISTIC_DIR = GENERATED_DIR / "linguistic_frame"

BATCH_SIZE = 2**15
DS_SHUFFLE_BUFFER = 3
DS_PREFETCH = 5


def get_dataset(kind: str) -> tf.data.Dataset:
    """Get dataset."""
    assert kind in {"train", "val", "test"}

    acoustic_paths = sorted((ACOUSTIC_DIR / kind).glob("*"))
    linguistic_paths = sorted((LINGUISTIC_DIR / kind).glob("*"))

    acoustic_arrays = [np.load(path) for path in acoustic_paths]
    linguistic_arrays = [np.load(path) for path in linguistic_paths]

    # Truncate, same length
    # TODO: Do it in preprocess
    acoustic_arrays = [
        ac_arr[: len(ln_arr)]
        for ac_arr, ln_arr in zip(acoustic_arrays, linguistic_arrays)
    ]

    def generator(
        linguistic_arrays: List[np.ndarray], acoustic_arrays: List[np.ndarray]
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for lng, aco in zip(linguistic_arrays, acoustic_arrays):
            yield lng, aco

    whole_ds = tf.data.Dataset.from_generator(
        generator,
        args=(linguistic_arrays, acoustic_arrays),
        output_signature=(
            tf.TensorSpec(shape=(None, 329), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1027), dtype=tf.float32),
        ),
    )

    def convert(dataset: tf.data.Dataset) -> tf.data.Dataset:
        lng = dataset.map(lambda x, y: x).flat_map(tf.data.Dataset.from_tensor_slices)
        f0 = dataset.map(lambda x, y: y[:, 0:1]).flat_map(
            tf.data.Dataset.from_tensor_slices
        )
        sp = dataset.map(lambda x, y: y[:, 1:514]).flat_map(
            tf.data.Dataset.from_tensor_slices
        )
        ap = dataset.map(lambda x, y: y[:, 514:1027]).flat_map(
            tf.data.Dataset.from_tensor_slices
        )
        target = tf.data.Dataset.zip((f0, sp, ap))
        return tf.data.Dataset.zip((lng, target))

    dataset = (
        convert(whole_ds)
        .shuffle(BATCH_SIZE * DS_SHUFFLE_BUFFER)
        .batch(BATCH_SIZE)
        .prefetch(DS_PREFETCH)
    )
    return dataset


def _extend(ipt: tf.keras.Tensor, name: str) -> tf.keras.Tensor:

    resid_feature = Reshape((1, -1))(
        Dense(32, activation="relu", name=f"{name}_res_f")(ipt)
    )
    resid_space = Reshape((-1, 1))(
        Dense(16, activation="relu", name=f"{name}_res_s")(ipt)
    )

    x = Reshape((1, -1))(ipt)
    x = Conv1DTranspose(128, 7, strides=4, padding="same", name=f"{name}_c_1")(x)
    x = ReLU()(x)
    x = Conv1DTranspose(32, 7, strides=4, padding="same")(x)
    x = Add(name=f"{name}_add")([x, resid_feature, resid_space])
    x = ReLU()(x)
    x = Conv1DTranspose(8, 7, strides=4, padding="same")(x)
    x = ReLU()(x)
    x = Conv1DTranspose(2, 7, strides=4, padding="same")(x)
    x = ReLU()(x)
    x = Conv1DTranspose(1, 3, strides=2, padding="same")(x)
    x = Flatten()(x)
    last = Dense(1)(x)
    x = Concatenate(name=f"{name}_out")([x, last])
    return x


def get_model(input_dim=329):
    """Get model. (None, 319) -> ((None, 1), (None, 513), (None, 513))"""
    ipt = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(
        256,
    )(ipt)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(
        256,
    )(x)
    x = tf.keras.layers.ReLU()(x)

    sp_out = _extend(x, "sp")
    ap_out = _extend(x, "ap")

    x = tf.keras.layers.Dense(
        64,
    )(x)
    x = tf.keras.layers.ReLU()(x)

    f0_out = tf.keras.layers.Dense(1, name="f0")(x)

    model = tf.keras.models.Model(inputs=ipt, outputs=(f0_out, sp_out, ap_out))
    return model


model = get_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=(MSE(), MSE(), MSE()),
)


train_ds = get_dataset("train")
val_ds = get_dataset("val")
model.fit(train_ds, validation_data=val_ds, epochs=1000)

model.save(GENERATED_DIR / "acoustic_model.h5")
