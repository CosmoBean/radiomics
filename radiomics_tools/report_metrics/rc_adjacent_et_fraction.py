"""Fraction of enhancing tumor that directly borders the resection cavity."""

from __future__ import annotations

import numpy as np

from .intensity import rc_adjacent_et_fraction as _rc_adjacent_et_fraction


def rc_adjacent_et_fraction(
    mask_array: np.ndarray,
    *,
    iterations: int = 1,
    connectivity: int = 1,
) -> float:
    """Return the fraction of ET voxels adjacent to RC.

    Adjacency is computed by dilating the RC mask and measuring the share of ET voxels
    that fall within that one-step neighborhood.

    Args:
        mask_array: 3D multiclass segmentation with ET as `1` and RC as `4`.
        iterations: Dilation radius in voxel steps.
        connectivity: Binary-connectivity passed to `generate_binary_structure`.

    Returns:
        A fraction between `0` and `1`, or `nan` if ET is absent.
    """

    return _rc_adjacent_et_fraction(mask_array, iterations=iterations, connectivity=connectivity)

