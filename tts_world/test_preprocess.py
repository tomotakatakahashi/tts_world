"""Unit tests."""

import numpy as np
import preprocess as pp

# pylint: disable=missing-function-docstring


def test_statistics_axis():
    def _correct(arrays, func):
        """Correct impl for reference."""
        arrays_concat = np.concatenate(arrays, axis=0)
        return func(arrays_concat, axis=0)

    arrays = [np.array([[0, 1]]), np.array([[1, 2], [2, 3]])]
    assert np.allclose(pp.statistics_axis(arrays, np.mean), _correct(arrays, np.mean))
    assert np.allclose(pp.statistics_axis(arrays, np.std), _correct(arrays, np.std))
