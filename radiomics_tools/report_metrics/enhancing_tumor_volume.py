"""Enhancing tumor volume from the multiclass segmentation mask."""

from __future__ import annotations

import numpy as np

from .volumes import enhancing_tumor_volume_cc as _enhancing_tumor_volume_cc


def enhancing_tumor_volume_cc(mask_array: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Return enhancing tumor volume in cubic centimeters.

    Args:
        mask_array: 3D multiclass segmentation with ET encoded as label `1`.
        spacing: Voxel spacing in millimeters as `(x, y, z)`.

    Returns:
        The ET compartment volume in cubic centimeters.
    """

    return _enhancing_tumor_volume_cc(mask_array, spacing)

