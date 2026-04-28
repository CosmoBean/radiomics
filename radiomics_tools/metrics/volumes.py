"""Compartment volume helpers for multiclass tumor masks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .constants import LABEL_MAP, WHOLE_TUMOR_LABELS


def _normalize_labels(labels: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(labels, int):
        return (labels,)
    return tuple(int(label) for label in labels)


def _as_mask_array(mask_array: np.ndarray) -> np.ndarray:
    data = np.asarray(mask_array)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D mask array, got shape {data.shape}")
    return data


def voxel_volume_cc(spacing: tuple[float, float, float]) -> float:
    if len(spacing) < 3:
        raise ValueError(f"Expected 3 spacing values, got {spacing}")
    return float(spacing[0] * spacing[1] * spacing[2]) / 1000.0


def mask_for_labels(mask_array: np.ndarray, labels: int | Iterable[int]) -> np.ndarray:
    data = _as_mask_array(mask_array)
    return np.isin(data, _normalize_labels(labels))


def voxel_count_for_labels(mask_array: np.ndarray, labels: int | Iterable[int]) -> int:
    return int(mask_for_labels(mask_array, labels).sum())


def volume_cc_for_labels(
    mask_array: np.ndarray,
    spacing: tuple[float, float, float],
    labels: int | Iterable[int],
) -> float:
    return float(voxel_count_for_labels(mask_array, labels) * voxel_volume_cc(spacing))


def enhancing_tumor_volume_cc(mask_array: np.ndarray, spacing: tuple[float, float, float]) -> float:
    return volume_cc_for_labels(mask_array, spacing, LABEL_MAP["et"])


def non_enhancing_tumor_core_volume_cc(mask_array: np.ndarray, spacing: tuple[float, float, float]) -> float:
    return volume_cc_for_labels(mask_array, spacing, LABEL_MAP["netc"])


def snfh_volume_cc(mask_array: np.ndarray, spacing: tuple[float, float, float]) -> float:
    return volume_cc_for_labels(mask_array, spacing, LABEL_MAP["snhf"])


def resection_cavity_volume_cc(mask_array: np.ndarray, spacing: tuple[float, float, float]) -> float:
    return volume_cc_for_labels(mask_array, spacing, LABEL_MAP["rc"])


def whole_tumor_volume_cc(mask_array: np.ndarray, spacing: tuple[float, float, float]) -> float:
    return volume_cc_for_labels(mask_array, spacing, WHOLE_TUMOR_LABELS)


def tumor_compartment_volumes_cc(
    mask_array: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, float]:
    return {
        "enhancing_tumor_volume_cc": enhancing_tumor_volume_cc(mask_array, spacing),
        "non_enhancing_tumor_core_volume_cc": non_enhancing_tumor_core_volume_cc(mask_array, spacing),
        "snhf_volume_cc": snfh_volume_cc(mask_array, spacing),
        "resection_cavity_volume_cc": resection_cavity_volume_cc(mask_array, spacing),
        "whole_tumor_volume_cc": whole_tumor_volume_cc(mask_array, spacing),
    }

