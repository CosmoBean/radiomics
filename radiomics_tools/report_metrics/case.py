"""Case-level wrappers that bundle the report metrics into tool-call outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bidimensional_product import enhancing_tumor_bidimensional_product_cm2
from .enhancing_tumor_volume import enhancing_tumor_volume_cc
from .io import load_case_data
from .mean_flair_intensity_within_snhf import mean_flair_intensity_within_snhf
from .non_enhancing_tumor_core_volume import non_enhancing_tumor_core_volume_cc
from .rc_adjacent_et_fraction import rc_adjacent_et_fraction
from .resection_cavity_volume import resection_cavity_volume_cc
from .snhf_volume import snhf_volume_cc
from .t1ce_to_t1_intensity_ratio_within_et import t1ce_to_t1_intensity_ratio_within_et
from .whole_tumor_volume import whole_tumor_volume_cc


@dataclass(frozen=True)
class CaseMetricPaths:
    mask_path: Path
    t1_path: Path | None = None
    t1ce_path: Path | None = None
    flair_path: Path | None = None


def compute_case_report_metrics(
    *,
    mask_array: np.ndarray,
    spacing: tuple[float, float, float],
    t1_array: np.ndarray | None = None,
    t1ce_array: np.ndarray | None = None,
    flair_array: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "enhancing_tumor_volume_cc": enhancing_tumor_volume_cc(mask_array, spacing),
        "non_enhancing_tumor_core_volume_cc": non_enhancing_tumor_core_volume_cc(mask_array, spacing),
        "snhf_volume_cc": snhf_volume_cc(mask_array, spacing),
        "resection_cavity_volume_cc": resection_cavity_volume_cc(mask_array, spacing),
        "whole_tumor_volume_cc": whole_tumor_volume_cc(mask_array, spacing),
    }
    metrics["bidimensional_product_cm2"] = enhancing_tumor_bidimensional_product_cm2(mask_array, spacing)
    metrics["rc_adjacent_et_fraction"] = rc_adjacent_et_fraction(mask_array)
    metrics["t1ce_to_t1_intensity_ratio_within_et"] = float("nan")
    if t1_array is not None and t1ce_array is not None:
        metrics["t1ce_to_t1_intensity_ratio_within_et"] = t1ce_to_t1_intensity_ratio_within_et(
            t1_array,
            t1ce_array,
            mask_array,
        )
    metrics["mean_flair_intensity_within_snhf"] = float("nan")
    if flair_array is not None:
        metrics["mean_flair_intensity_within_snhf"] = mean_flair_intensity_within_snhf(flair_array, mask_array)
    return metrics


def compute_case_report_metrics_from_paths(paths: CaseMetricPaths) -> dict[str, float]:
    loaded = load_case_data(
        mask_path=paths.mask_path,
        t1_path=paths.t1_path,
        t1ce_path=paths.t1ce_path,
        flair_path=paths.flair_path,
    )
    return compute_case_report_metrics(
        mask_array=loaded.mask_array,
        spacing=loaded.spacing,
        t1_array=loaded.t1_array,
        t1ce_array=loaded.t1ce_array,
        flair_array=loaded.flair_array,
    )
