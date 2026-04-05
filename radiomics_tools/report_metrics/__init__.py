"""Tool-call friendly report metrics for postoperative glioma scans."""

from .case import CaseMetricPaths, compute_case_report_metrics, compute_case_report_metrics_from_paths
from .geometry import bidimensional_product_cm2, enhancing_tumor_bidimensional_product_cm2
from .intensity import (
    mean_flair_intensity_within_snhf,
    mean_intensity_for_labels,
    rc_adjacent_et_fraction,
    t1ce_to_t1_intensity_ratio_within_et,
)
from .volumes import (
    enhancing_tumor_volume_cc,
    non_enhancing_tumor_core_volume_cc,
    resection_cavity_volume_cc,
    snfh_volume_cc,
    tumor_compartment_volumes_cc,
    whole_tumor_volume_cc,
)

__all__ = [
    "CaseMetricPaths",
    "bidimensional_product_cm2",
    "compute_case_report_metrics",
    "compute_case_report_metrics_from_paths",
    "enhancing_tumor_bidimensional_product_cm2",
    "enhancing_tumor_volume_cc",
    "mean_flair_intensity_within_snhf",
    "mean_intensity_for_labels",
    "non_enhancing_tumor_core_volume_cc",
    "rc_adjacent_et_fraction",
    "resection_cavity_volume_cc",
    "snhf_volume_cc",
    "t1ce_to_t1_intensity_ratio_within_et",
    "tumor_compartment_volumes_cc",
    "whole_tumor_volume_cc",
]

