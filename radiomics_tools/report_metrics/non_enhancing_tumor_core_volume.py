"""Non-enhancing tumor core volume from the multiclass segmentation mask."""

from __future__ import annotations

import numpy as np

from .volumes import non_enhancing_tumor_core_volume_cc as _netc_volume_cc


def non_enhancing_tumor_core_volume_cc(mask_array: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Return NETC volume in cubic centimeters.

    Args:
        mask_array: 3D multiclass segmentation with NETC encoded as label `2`.
        spacing: Voxel spacing in millimeters as `(x, y, z)`.

    Returns:
        The non-enhancing tumor core volume in cubic centimeters.
    """

    return _netc_volume_cc(mask_array, spacing)

