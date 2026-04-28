"""Resection cavity volume from the multiclass segmentation mask."""

from __future__ import annotations

import numpy as np

from .volumes import resection_cavity_volume_cc as _resection_cavity_volume_cc


def resection_cavity_volume_cc(mask_array: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Return RC volume in cubic centimeters.

    Args:
        mask_array: 3D multiclass segmentation with RC encoded as label `4`.
        spacing: Voxel spacing in millimeters as `(x, y, z)`.

    Returns:
        The resection cavity volume in cubic centimeters.
    """

    return _resection_cavity_volume_cc(mask_array, spacing)

