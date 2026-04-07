"""Mean FLAIR intensity measured within the SNFH compartment."""

from __future__ import annotations

import numpy as np

from .intensity import mean_flair_intensity_within_snhf as _mean_flair_intensity_within_snhf


def mean_flair_intensity_within_snhf(flair_array: np.ndarray, mask_array: np.ndarray) -> float:
    """Return mean FLAIR signal inside the SNFH compartment.

    Args:
        flair_array: 3D FLAIR volume aligned to the segmentation.
        mask_array: 3D multiclass segmentation with SNFH encoded as label `3`.

    Returns:
        The mean FLAIR intensity across SNFH voxels.
    """

    return _mean_flair_intensity_within_snhf(flair_array, mask_array)
