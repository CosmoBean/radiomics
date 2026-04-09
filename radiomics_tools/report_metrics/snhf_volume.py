"""SNFH volume from the multiclass segmentation mask."""

from __future__ import annotations

import numpy as np

from .volumes import snfh_volume_cc as _snfh_volume_cc


def snhf_volume_cc(mask_array: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Return SNFH volume in cubic centimeters.

    Args:
        mask_array: 3D multiclass segmentation with SNFH encoded as label `3`.
        spacing: Voxel spacing in millimeters as `(x, y, z)`.

    Returns:
        The surrounding non-enhancing FLAIR hyperintensity volume in cubic centimeters.
    """

    return _snfh_volume_cc(mask_array, spacing)
