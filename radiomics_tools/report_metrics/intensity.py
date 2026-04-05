"""Signal-based report metrics derived from labeled tumor compartments."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from .constants import EPSILON, LABEL_MAP
from .volumes import mask_for_labels


def _as_image_array(image_array: np.ndarray) -> np.ndarray:
    data = np.asarray(image_array, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image array, got shape {data.shape}")
    return data


def _require_same_shape(image_array: np.ndarray, mask_array: np.ndarray) -> None:
    if image_array.shape != mask_array.shape:
        raise ValueError(
            f"Image and mask shape mismatch: image {image_array.shape}, mask {mask_array.shape}"
        )


def mean_intensity_in_mask(image_array: np.ndarray, binary_mask: np.ndarray) -> float:
    voxels = image_array[binary_mask]
    if voxels.size == 0:
        return float("nan")
    return float(np.mean(voxels))


def mean_intensity_for_labels(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    labels: int | tuple[int, ...],
) -> float:
    image = _as_image_array(image_array)
    mask = np.asarray(mask_array)
    _require_same_shape(image, mask)
    return mean_intensity_in_mask(image, mask_for_labels(mask, labels))


def t1ce_to_t1_intensity_ratio_within_et(
    t1_array: np.ndarray,
    t1ce_array: np.ndarray,
    mask_array: np.ndarray,
) -> float:
    et_mean_t1 = mean_intensity_for_labels(t1_array, mask_array, LABEL_MAP["et"])
    et_mean_t1ce = mean_intensity_for_labels(t1ce_array, mask_array, LABEL_MAP["et"])
    if not np.isfinite(et_mean_t1) or not np.isfinite(et_mean_t1ce):
        return float("nan")
    return float(et_mean_t1ce / max(abs(et_mean_t1), EPSILON))


def rc_adjacent_et_fraction(
    mask_array: np.ndarray,
    *,
    iterations: int = 1,
    connectivity: int = 1,
) -> float:
    mask = np.asarray(mask_array)
    et_mask = mask_for_labels(mask, LABEL_MAP["et"])
    rc_mask = mask_for_labels(mask, LABEL_MAP["rc"])
    if not et_mask.any():
        return float("nan")
    rc_adjacent = binary_dilation(
        rc_mask,
        structure=generate_binary_structure(3, connectivity),
        iterations=iterations,
    )
    return float(np.logical_and(et_mask, rc_adjacent).sum() / max(1, et_mask.sum()))


def mean_flair_intensity_within_snhf(flair_array: np.ndarray, mask_array: np.ndarray) -> float:
    return mean_intensity_for_labels(flair_array, mask_array, LABEL_MAP["snhf"])

