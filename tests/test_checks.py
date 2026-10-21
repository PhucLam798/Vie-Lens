"""Tests for dtype-robust boolean mask comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vilens.checks import count_mask_disagreements


def test_numpy_and_nullable_boolean_equal_masks() -> None:
    """Treat equal NumPy and pandas nullable boolean values as equal."""
    left = np.array([True, False, True], dtype=np.bool_)
    right = pd.Series([True, False, True], dtype="boolean")
    assert count_mask_disagreements(left, right) == 0


def test_one_differing_mask_element() -> None:
    """Count one differing position across boolean dtypes."""
    left = np.array([True, False, True], dtype=bool)
    right = pd.Series([True, True, True], dtype="boolean")
    assert count_mask_disagreements(left, right) == 1


def test_nullable_boolean_na_fails() -> None:
    """Reject nullable boolean masks that contain pandas.NA."""
    left = pd.Series([True, pd.NA], dtype="boolean")
    right = np.array([True, False], dtype=bool)
    with pytest.raises(ValueError, match="contains NA"):
        count_mask_disagreements(left, right)


def test_different_mask_lengths_fail() -> None:
    """Reject masks with unequal lengths."""
    left = np.array([True, False], dtype=bool)
    right = pd.Series([True], dtype="boolean")
    with pytest.raises(ValueError, match="lengths differ"):
        count_mask_disagreements(left, right)
