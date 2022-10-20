"""Train the duration model."""

from pathlib import Path

import numpy as np
import tensorflow as tf

GENERATED_DIR = Path.cwd() / "generated"

DURATION_DIR = GENERATED_DIR / "duration"
LINGUISTIC_DIR = GENERATED_DIR / "linguistic"

DATA_SIZE = 4000

duration_paths = sorted(DURATION_DIR.glob("*"))[
    :DATA_SIZE
]  # Remove mean.npy and std.npy
linguistic_paths = sorted(LINGUISTIC_DIR.glob("*"))[:DATA_SIZE]

duration_arrays = [np.load(path) for path in duration_paths]
linguistic_arrays = [np.load(path) for path in linguistic_paths]

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            64,
        ),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(
            64,
        ),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(
            64,
        ),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1, activation=None),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["MAE"],
)

BATCH_SIZE = 2**13

dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            np.concatenate(linguistic_arrays, axis=0),
            np.concatenate(duration_arrays, axis=0),
        )
    )
    .shuffle(BATCH_SIZE * 10)
    .batch(BATCH_SIZE)
    .prefetch(10)
)
model.fit(dataset, epochs=100)

model.save("duration_model.h5")
