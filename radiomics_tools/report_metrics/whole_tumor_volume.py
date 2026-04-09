"""Whole tumor volume from the multiclass segmentation mask."""

from __future__ import annotations

import numpy as np

from .volumes import whole_tumor_volume_cc as _whole_tumor_volume_cc


def whole_tumor_volume_cc(mask_array: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Return whole-tumor volume in cubic centimeters.

    Args:
        mask_array: 3D multiclass segmentation with tumor labels `1/2/3/4`.
        spacing: Voxel spacing in millimeters as `(x, y, z)`.

    Returns:
        The union volume across ET, NETC, SNFH, and RC in cubic centimeters.
    """

    return _whole_tumor_volume_cc(mask_array, spacing)

