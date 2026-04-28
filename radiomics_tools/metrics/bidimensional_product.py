"""Bidimensional product feature for enhancing tumor burden."""

from __future__ import annotations

import numpy as np

from .geometry import enhancing_tumor_bidimensional_product_cm2 as _et_bidimensional_product_cm2


def enhancing_tumor_bidimensional_product_cm2(
    mask_array: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """Return the maximum ET bidimensional product in square centimeters.

    The function scans axial slices and returns the largest in-plane extent product
    for the enhancing tumor compartment.

    Args:
        mask_array: 3D multiclass segmentation with ET encoded as label `1`.
        spacing: Voxel spacing in millimeters as `(x, y, z)`.

    Returns:
        The maximum ET 2D size proxy in square centimeters.
    """

    return _et_bidimensional_product_cm2(mask_array, spacing)

