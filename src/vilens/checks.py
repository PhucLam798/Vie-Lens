"""Dtype-robust comparison helpers for boolean checks."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

MaskInput: TypeAlias = np.ndarray | pd.Series


def _as_boolean_array(value: MaskInput, name: str) -> np.ndarray:
    """Validate a one-dimensional boolean mask with no missing values."""
    if not isinstance(value, (np.ndarray, pd.Series)):
        raise ValueError(f"{name} must be a NumPy array or pandas Series")
    if not is_bool_dtype(value.dtype):
        raise ValueError(f"{name} must have a boolean dtype, got {value.dtype}")
    if isinstance(value, np.ma.MaskedArray) and bool(np.ma.getmaskarray(value).any()):
        raise ValueError(f"{name} contains NA values")
    missing = np.asarray(pd.isna(value), dtype=bool)
    if bool(missing.any()):
        raise ValueError(f"{name} contains NA values")
    array = np.asarray(value, dtype=bool)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def count_mask_disagreements(a: MaskInput, b: MaskInput) -> int:
    """Return the number of positions that differ between two non-null boolean masks."""
    array_a = _as_boolean_array(a, "a")
    array_b = _as_boolean_array(b, "b")
    if len(array_a) != len(array_b):
        raise ValueError(f"mask lengths differ: {len(array_a)} != {len(array_b)}")
    return int(np.count_nonzero(array_a != array_b))
