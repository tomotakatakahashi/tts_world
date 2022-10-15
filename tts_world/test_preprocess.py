"""Unit tests."""

import numpy as np
import preprocess as pp

# pylint: disable=missing-function-docstring


def test_statistics_axis():
    def _correct(arrays, func):
        """Correct impl for reference."""
        arrays_concat = np.concatenate(arrays, axis=0)
        return func(arrays_concat, axis=0)

    arrays = [np.arange(3), np.arange(100)]
    assert np.allclose(pp.statistics_axis(arrays, np.mean), _correct(arrays, np.mean))
    assert np.allclose(pp.statistics_axis(arrays, np.std), _correct(arrays, np.std))
