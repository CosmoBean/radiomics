"""Geometric burden features derived from compartment masks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .constants import LABEL_MAP
from .volumes import mask_for_labels


def bidimensional_product_cm2(
    mask_array: np.ndarray,
    spacing: tuple[float, float, float],
    labels: int | Iterable[int] = (LABEL_MAP["et"],),
) -> float:
    binary_mask = mask_for_labels(mask_array, labels)
    if not binary_mask.any():
        return float("nan")
    dy_mm = float(spacing[1]) if len(spacing) > 1 else 1.0
    dx_mm = float(spacing[0]) if len(spacing) > 0 else 1.0
    best = 0.0
    for slice_mask in binary_mask:
        if not slice_mask.any():
            continue
        coords = np.argwhere(slice_mask)
        y_extent_mm = (coords[:, 0].max() - coords[:, 0].min() + 1) * dy_mm
        x_extent_mm = (coords[:, 1].max() - coords[:, 1].min() + 1) * dx_mm
        best = max(best, (y_extent_mm / 10.0) * (x_extent_mm / 10.0))
    return float(best)


def enhancing_tumor_bidimensional_product_cm2(
    mask_array: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    return bidimensional_product_cm2(mask_array, spacing, labels=(LABEL_MAP["et"],))

