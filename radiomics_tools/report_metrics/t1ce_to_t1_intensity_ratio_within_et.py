"""T1CE-to-T1 signal ratio measured within enhancing tumor."""

from __future__ import annotations

import numpy as np

from .intensity import t1ce_to_t1_intensity_ratio_within_et as _t1ce_to_t1_intensity_ratio_within_et


def t1ce_to_t1_intensity_ratio_within_et(
    t1_array: np.ndarray,
    t1ce_array: np.ndarray,
    mask_array: np.ndarray,
) -> float:
    """Return the ET-restricted mean T1CE:T1 signal ratio.

    Args:
        t1_array: 3D T1 volume aligned to the mask.
        t1ce_array: 3D post-contrast T1 volume aligned to the mask.
        mask_array: 3D multiclass segmentation with ET encoded as label `1`.

    Returns:
        The mean T1CE intensity inside ET divided by the mean T1 intensity inside ET.
    """

    return _t1ce_to_t1_intensity_ratio_within_et(t1_array, t1ce_array, mask_array)

