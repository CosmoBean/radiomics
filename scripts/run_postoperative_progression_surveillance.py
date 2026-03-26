#!/usr/bin/env python3
"""Run the postoperative progression-surveillance radiomics pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, generate_binary_structure
from lightgbm import LGBMClassifier
from radiomics import featureextractor, logger as radiomics_logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional UI dependency
    tqdm = None

sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)


MODALITY_PATH_COLUMNS = {
    "t1": "native_t1_path",
    "t1c": "native_t1c_path",
    "flair": "native_flair_path",
    "t2": "native_t2_path",
}
ALL_MODALITIES = tuple(MODALITY_PATH_COLUMNS.keys())
UNION_LABELS = (1, 2, 3)
EPS = 1e-8
CLINICAL_FEATURE_PREFIX = "clin_"
ENGINEERED_FEATURE_PREFIX = "eng_"
CASE_ID_COLUMNS = ("patient_id", "timepoint", "timepoint_number")
CASE_METADATA_COLUMNS = (
    *CASE_ID_COLUMNS,
    "label",
    "clinical_progression",
    "progression_day",
    "days_from_diagnosis_to_mri",
    "delta_to_progression_days",
    "union_mask_voxels",
)
LATE_TREATMENT_START_COLUMNS = (
    "clinical_number_of_days_from_diagnosis_to_starting_2nd_additional_therapy",
    "clinical_number_of_days_from_diagnosis_to_start_immunotherapy",
    "clinical_days_from_diagnosis_to_new_treatment",
)
MOLECULAR_FEATURE_COLUMNS = (
    "clinical_idh1_mutation",
    "clinical_idh2_mutation",
    "clinical_1p_19q",
    "clinical_atrx_mutation",
    "clinical_mgmt_methylation",
    "clinical_tert_promoter_mutation",
    "clinical_egfr_amplification",
    "clinical_pten_mutation",
    "clinical_cdkn2a_b_deletion",
    "clinical_tp53_alteration",
    "clinical_chromosome_7_gain_and_chromosome_10_loss",
)
HYBRID_BASIC_EXTRA_COLUMNS = (
    "clinical_age_at_diagnosis",
    "clinical_sex_at_birth",
)
ENGINEERED_CATEGORICAL_COLUMNS = (
    "clinical_grade_of_primary_brain_tumor",
    "clinical_primary_diagnosis",
    "clinical_previous_brain_tumor",
    "clinical_grade_of_previous_brain_tumor",
)
REPORT_LABEL_MAP = {
    "et": 1,
    "netc": 2,
    "snhf": 3,
    "rc": 4,
}


@dataclass
class PreprocessorState:
    medians: pd.Series
    keep_columns: list[str]
    scale_mean: np.ndarray | None
    scale_scale: np.ndarray | None
    scale: bool
    variance_threshold: float
    corr_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the postoperative progression-surveillance radiomics pipeline on MU-Glioma-Post."
    )
    parser.add_argument(
        "--experiment-index",
        type=Path,
        default=Path("processed/manifests/experiment_index.csv"),
        help="Merged experiment index with clinical labels and native image paths.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root for resolving relative paths.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("processed/postoperative_progression_surveillance"),
        help="Cache directory for surveillance-specific preprocessed images and per-case features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/postoperative_progression_surveillance"),
        help="Directory for surveillance outputs.",
    )
    parser.add_argument(
        "--radiomics-yaml",
        type=Path,
        default=Path("configs/postoperative_progression_surveillance_radiomics.yaml"),
        help="PyRadiomics YAML configuration.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="post_progression",
        choices=["patient_progression", "post_progression", "within_window"],
        help="How to derive the timepoint label from the clinical columns.",
    )
    parser.add_argument(
        "--pre-progression-only",
        action="store_true",
        help="Exclude scans at or after first progression, keeping only non-progressor scans and scans strictly before progression.",
    )
    parser.add_argument(
        "--earliest-scan-only",
        action="store_true",
        help="Keep only the earliest usable scan per patient after cohort filtering.",
    )
    parser.add_argument(
        "--exclude-after-late-treatment",
        action="store_true",
        help="Exclude scans at or after the earliest recorded late-treatment start (2nd additional therapy, immunotherapy, or new treatment).",
    )
    parser.add_argument(
        "--progression-window-days",
        type=int,
        default=120,
        help="Window used when --label-mode=within_window.",
    )
    parser.add_argument(
        "--target-test-patients",
        type=int,
        default=30,
        help="Target number of held-out patients.",
    )
    parser.add_argument(
        "--target-test-samples",
        type=int,
        default=96,
        help="Target number of held-out samples.",
    )
    parser.add_argument(
        "--target-test-positives",
        type=int,
        default=53,
        help="Target number of held-out positive samples.",
    )
    parser.add_argument(
        "--split-search-iters",
        type=int,
        default=20000,
        help="Random-search iterations for approximating the target held-out cohort.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Parallel workers for case preprocessing and feature extraction.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="lightgbm",
        help="Comma-separated model families to search: lightgbm,logreg,rf,svm.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="all",
        help="Comma-separated modalities to model from t1,t1c,flair,t2. Use 'all' for every modality.",
    )
    parser.add_argument(
        "--clinical-feature-set",
        type=str,
        default="none",
        choices=[
            "none",
            "molecular",
            "hybrid_basic",
            "hybrid_engineered",
            "hybrid_engineered_biologic",
            "report_core",
            "report_timing",
            "report_full",
        ],
        help="Optional patient-level non-imaging features to merge into the model table.",
    )
    parser.add_argument(
        "--feature-subsets",
        type=str,
        default="512,256,128,64",
        help="Comma-separated top-k feature subset sizes after ranking.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=400,
        help="Optuna trials per model/subset.",
    )
    parser.add_argument(
        "--ranking-folds",
        type=int,
        default=10,
        help="Folds for the L1 + permutation consensus ranking stage.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Patient-aware CV folds for Optuna and OOF calibration.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=100,
        help="Permutation repeats for feature ranking.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Bootstrap iterations for the test ROC AUC interval.",
    )
    parser.add_argument(
        "--lightgbm-device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "cuda"],
        help="Device type for LightGBM trials. 'gpu' uses the OpenCL GPU trainer when available.",
    )
    parser.add_argument(
        "--progress-bar",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Whether to render tqdm progress bars for interactive runs.",
    )
    parser.add_argument(
        "--feature-table",
        type=Path,
        default=None,
        help="Optional precomputed radiomics feature table to reuse instead of rebuilding from per-case caches.",
    )
    parser.add_argument(
        "--test-patients-file",
        type=Path,
        default=None,
        help="Optional newline-delimited held-out patient list to reuse instead of sampling a new split.",
    )
    parser.add_argument(
        "--reuse-case-features",
        action="store_true",
        help="Reuse cached per-case feature JSON files if present.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on the number of labeled cases, for smoke testing.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_progress(output_dir: Path, payload: dict[str, object]) -> None:
    write_json(output_dir / "progress.json", payload)


def progress_bar_enabled(args: argparse.Namespace) -> bool:
    if args.progress_bar == "on":
        return tqdm is not None
    if args.progress_bar == "off":
        return False
    return tqdm is not None and sys.stderr.isatty()


def resolve_repo_path(repo_root: Path, value: object) -> Path:
    return (repo_root / str(value)).resolve()


def parse_modalities(value: str) -> list[str]:
    requested = [token.strip().lower() for token in str(value).split(",") if token.strip()]
    if not requested or requested == ["all"]:
        return list(ALL_MODALITIES)
    invalid = sorted(set(requested) - set(ALL_MODALITIES))
    if invalid:
        raise ValueError(f"Unsupported modalities: {invalid}. Choose from {list(ALL_MODALITIES)} or 'all'.")
    ordered = [modality for modality in ALL_MODALITIES if modality in requested]
    if not ordered:
        raise ValueError("At least one modality must be selected.")
    return ordered


def validate_args(args: argparse.Namespace) -> None:
    if args.pre_progression_only and args.label_mode == "post_progression":
        raise ValueError(
            "--pre-progression-only is incompatible with --label-mode post_progression. "
            "Use --label-mode within_window or patient_progression for forward prediction."
        )


def case_metadata_record(case: dict[str, object], union_voxels: int | None = None) -> dict[str, object]:
    union_value = union_voxels
    if union_value is None:
        union_value = case.get("union_voxels", case.get("union_mask_voxels", 0))
    return {
        "patient_id": str(case["patient_id"]),
        "timepoint": str(case["timepoint"]),
        "timepoint_number": int(case["timepoint_number"]),
        "label": int(case["label"]),
        "clinical_progression": int(case["clinical_progression"]),
        "progression_day": float(case["progression_day"]) if pd.notna(case["progression_day"]) else None,
        "days_from_diagnosis_to_mri": float(case["days_from_diagnosis_to_mri"])
        if pd.notna(case["days_from_diagnosis_to_mri"])
        else None,
        "delta_to_progression_days": float(case["delta_to_progression_days"])
        if pd.notna(case["delta_to_progression_days"])
        else None,
        "union_mask_voxels": int(union_value),
    }


def case_metadata_frame(cases: pd.DataFrame) -> pd.DataFrame:
    renamed = cases.rename(columns={"union_voxels": "union_mask_voxels"})
    columns = [
        "patient_id",
        "timepoint",
        "timepoint_number",
        "label",
        "clinical_progression",
        "progression_day",
        "days_from_diagnosis_to_mri",
        "delta_to_progression_days",
        "union_mask_voxels",
    ]
    return renamed[columns].copy()


def earliest_nonnegative_day(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index, dtype=float)
    values = df[available].apply(pd.to_numeric, errors="coerce")
    values = values.where(values >= 0)
    return values.min(axis=1, skipna=True)


def clean_categorical_codes(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        labels = numeric.astype("Int64").astype(str)
    else:
        labels = series.fillna("missing").astype(str).str.strip()
        labels = labels.replace({"": "missing", "nan": "missing", "None": "missing"})
    return labels.fillna("missing")


def add_engineered_numeric_feature(clinical: pd.DataFrame, name: str, values: pd.Series) -> None:
    clinical[f"{ENGINEERED_FEATURE_PREFIX}{name}"] = pd.to_numeric(values, errors="coerce").astype(float)


def volume_cc(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype(float) / 1000.0


def report_feature_cache_path(cache_root: Path, patient_id: str, timepoint: str) -> Path:
    return cache_root / "report_features" / patient_id / f"{timepoint}.json"


def compute_bidimensional_product_cm2(mask_array: np.ndarray, spacing: tuple[float, float, float]) -> float:
    if mask_array.ndim != 3:
        return float("nan")
    dy_mm = float(spacing[1]) if len(spacing) > 1 else 1.0
    dx_mm = float(spacing[0]) if len(spacing) > 0 else 1.0
    best = 0.0
    for slice_mask in mask_array:
        if not slice_mask.any():
            continue
        coords = np.argwhere(slice_mask)
        y_extent_mm = (coords[:, 0].max() - coords[:, 0].min() + 1) * dy_mm
        x_extent_mm = (coords[:, 1].max() - coords[:, 1].min() + 1) * dx_mm
        best = max(best, (y_extent_mm / 10.0) * (x_extent_mm / 10.0))
    return float(best)


def mean_signal_in_mask(image_array: np.ndarray, mask_array: np.ndarray) -> float:
    voxels = image_array[mask_array]
    if voxels.size == 0:
        return float("nan")
    return float(np.mean(voxels))


def compute_report_imaging_features_for_case(
    case: dict[str, object],
    repo_root: Path,
    cache_root: Path,
) -> dict[str, float]:
    patient_id = str(case["patient_id"])
    timepoint = str(case["timepoint"])
    cache_path = report_feature_cache_path(cache_root, patient_id, timepoint)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    mask_path = resolve_repo_path(repo_root, case["multiclass_mask_path"])
    mask_img = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    mask_array = sitk.GetArrayFromImage(mask_img)
    spacing = mask_img.GetSpacing()

    et_mask = mask_array == REPORT_LABEL_MAP["et"]
    snhf_mask = mask_array == REPORT_LABEL_MAP["snhf"]
    rc_mask = mask_array == REPORT_LABEL_MAP["rc"]

    t1_array = sitk.GetArrayFromImage(resolve_and_read_image(repo_root, case["native_t1_path"])).astype(np.float32, copy=False)
    t1c_array = sitk.GetArrayFromImage(resolve_and_read_image(repo_root, case["native_t1c_path"])).astype(np.float32, copy=False)
    flair_array = sitk.GetArrayFromImage(resolve_and_read_image(repo_root, case["native_flair_path"])).astype(np.float32, copy=False)

    et_mean_t1 = mean_signal_in_mask(t1_array, et_mask)
    et_mean_t1c = mean_signal_in_mask(t1c_array, et_mask)
    flair_mean_snhf = mean_signal_in_mask(flair_array, snhf_mask)

    adjacency_structure = generate_binary_structure(3, 1)
    rc_adjacent_et_fraction = float("nan")
    if et_mask.any():
        rc_adjacent = binary_dilation(rc_mask, structure=adjacency_structure, iterations=1)
        rc_adjacent_et_fraction = float(np.logical_and(et_mask, rc_adjacent).sum() / max(1, et_mask.sum()))

    features = {
        "bd_product_cm2": compute_bidimensional_product_cm2(et_mask, spacing),
        "t1ce_t1_signal_ratio_within_et": float(et_mean_t1c / max(abs(et_mean_t1), EPS))
        if np.isfinite(et_mean_t1c) and np.isfinite(et_mean_t1)
        else float("nan"),
        "rc_adjacent_et_fraction": rc_adjacent_et_fraction,
        "mean_flair_signal_within_snhf": flair_mean_snhf,
    }
    cache_path.write_text(json.dumps(features, sort_keys=True), encoding="utf-8")
    return features


def resolve_and_read_image(repo_root: Path, value: object) -> sitk.Image:
    return sitk.ReadImage(str(resolve_repo_path(repo_root, value)), sitk.sitkFloat32)


def add_report_volume_features(clinical: pd.DataFrame, cases: pd.DataFrame) -> None:
    for label_name, feature_name in (
        ("label1_voxels", "et_volume_cc"),
        ("label2_voxels", "netc_volume_cc"),
        ("label3_voxels", "snhf_volume_cc"),
        ("label4_voxels", "rc_volume_cc"),
        ("whole_tumor_voxels", "whole_tumor_volume_cc"),
    ):
        if label_name in cases.columns:
            add_engineered_numeric_feature(clinical, feature_name, volume_cc(cases[label_name]))


def add_report_timing_features(clinical: pd.DataFrame, cases: pd.DataFrame) -> None:
    if "days_from_diagnosis_to_mri" in cases.columns:
        mri_days = pd.to_numeric(cases["days_from_diagnosis_to_mri"], errors="coerce")
        add_engineered_numeric_feature(
            clinical,
            "days_from_diagnosis_to_current_mri",
            mri_days.where(mri_days >= 0),
        )
    if "clinical_number_of_days_from_diagnosis_to_radiation_therapy_end_date" in cases.columns and "days_from_diagnosis_to_mri" in cases.columns:
        radiation_end = pd.to_numeric(
            cases["clinical_number_of_days_from_diagnosis_to_radiation_therapy_end_date"],
            errors="coerce",
        )
        current_mri = pd.to_numeric(cases["days_from_diagnosis_to_mri"], errors="coerce")
        days_post_rt = current_mri - radiation_end
        days_post_rt = days_post_rt.where(days_post_rt >= 0)
        add_engineered_numeric_feature(clinical, "days_post_radiation_therapy_end", days_post_rt)


def add_report_imaging_features(clinical: pd.DataFrame, cases: pd.DataFrame, repo_root: Path, cache_root: Path) -> None:
    rows: list[dict[str, object]] = []
    for case in cases.to_dict(orient="records"):
        payload = {
            **{key: case[key] for key in CASE_ID_COLUMNS},
            **compute_report_imaging_features_for_case(case, repo_root=repo_root, cache_root=cache_root),
        }
        rows.append(payload)
    derived = pd.DataFrame(rows)
    for column in [col for col in derived.columns if col not in CASE_ID_COLUMNS]:
        clinical[f"{ENGINEERED_FEATURE_PREFIX}{column}"] = pd.to_numeric(derived[column], errors="coerce").astype(float)


def build_clinical_feature_frame(
    cases: pd.DataFrame,
    feature_set: str,
    repo_root: Path,
    cache_root: Path,
) -> pd.DataFrame:
    clinical = cases[list(CASE_ID_COLUMNS)].copy()
    if feature_set == "none":
        return clinical

    frames: list[pd.DataFrame] = []
    include_basic = feature_set in {
        "hybrid_basic",
        "hybrid_engineered",
        "hybrid_engineered_biologic",
        "report_core",
        "report_timing",
        "report_full",
    }
    include_engineered = feature_set in {"hybrid_engineered", "hybrid_engineered_biologic"}
    include_timing_engineering = feature_set == "hybrid_engineered"
    include_report_core = feature_set in {"report_core", "report_timing", "report_full"}
    include_report_timing = feature_set in {"report_timing", "report_full"}
    include_report_full = feature_set == "report_full"

    if include_basic and "clinical_age_at_diagnosis" in cases.columns:
        age = pd.to_numeric(cases["clinical_age_at_diagnosis"], errors="coerce")
        clinical[f"{CLINICAL_FEATURE_PREFIX}age_at_diagnosis"] = age.astype(float)
    if include_basic and "clinical_sex_at_birth" in cases.columns:
        sex = clean_categorical_codes(cases["clinical_sex_at_birth"])
        frames.append(
            pd.get_dummies(
                sex,
                prefix=f"{CLINICAL_FEATURE_PREFIX}sex_at_birth",
                prefix_sep="__",
                dtype=float,
            )
        )

    if include_engineered:
        union_voxels = (
            pd.to_numeric(cases["union_voxels"], errors="coerce")
            if "union_voxels" in cases.columns
            else pd.Series(np.nan, index=cases.index, dtype=float)
        )
        if union_voxels.notna().any():
            add_engineered_numeric_feature(clinical, "log_union_voxels", np.log1p(union_voxels.clip(lower=0)))
            add_engineered_numeric_feature(clinical, "sqrt_union_voxels", np.sqrt(union_voxels.clip(lower=0)))

        whole_tumor = (
            pd.to_numeric(cases["whole_tumor_voxels"], errors="coerce")
            if "whole_tumor_voxels" in cases.columns
            else pd.Series(np.nan, index=cases.index, dtype=float)
        )
        if whole_tumor.notna().any():
            add_engineered_numeric_feature(clinical, "log_whole_tumor_voxels", np.log1p(whole_tumor.clip(lower=0)))

        denom = union_voxels.replace(0, np.nan)
        for label_name, suffix in (
            ("label1_voxels", "label1_fraction"),
            ("label2_voxels", "label2_fraction"),
            ("label3_voxels", "label3_fraction"),
        ):
            if label_name not in cases.columns:
                continue
            label_values = pd.to_numeric(cases[label_name], errors="coerce")
            add_engineered_numeric_feature(clinical, suffix, label_values / denom)
            add_engineered_numeric_feature(clinical, f"log_{label_name}", np.log1p(label_values.clip(lower=0)))

        if include_timing_engineering and "days_from_diagnosis_to_mri" in cases.columns:
            mri_days = pd.to_numeric(cases["days_from_diagnosis_to_mri"], errors="coerce")
            add_engineered_numeric_feature(clinical, "log_days_from_diagnosis_to_mri", np.log1p(mri_days.clip(lower=0)))

        if include_timing_engineering and "timepoint_number" in cases.columns:
            add_engineered_numeric_feature(clinical, "timepoint_number", cases["timepoint_number"])

        for column in ENGINEERED_CATEGORICAL_COLUMNS:
            if column not in cases.columns:
                continue
            encoded = clean_categorical_codes(cases[column])
            short_name = column.removeprefix("clinical_")
            frames.append(
                pd.get_dummies(
                    encoded,
                    prefix=f"{CLINICAL_FEATURE_PREFIX}{short_name}",
                    prefix_sep="__",
                    dtype=float,
                )
            )

    if include_report_core:
        add_report_volume_features(clinical, cases)
        if "clinical_grade_of_primary_brain_tumor" in cases.columns:
            grade = clean_categorical_codes(cases["clinical_grade_of_primary_brain_tumor"])
            frames.append(
                pd.get_dummies(
                    grade,
                    prefix=f"{CLINICAL_FEATURE_PREFIX}grade_of_primary_brain_tumor",
                    prefix_sep="__",
                    dtype=float,
                )
            )

    if include_report_timing:
        add_report_timing_features(clinical, cases)

    if include_report_full:
        add_report_imaging_features(clinical, cases, repo_root=repo_root, cache_root=cache_root)

    for column in MOLECULAR_FEATURE_COLUMNS:
        if column not in cases.columns:
            continue
        encoded = clean_categorical_codes(cases[column])
        short_name = column.removeprefix("clinical_")
        frames.append(
            pd.get_dummies(
                encoded,
                prefix=f"{CLINICAL_FEATURE_PREFIX}{short_name}",
                prefix_sep="__",
                dtype=float,
            )
        )

    if frames:
        clinical = pd.concat([clinical, *frames], axis=1)
    return clinical


def merge_clinical_features(
    features: pd.DataFrame,
    cases: pd.DataFrame,
    feature_set: str,
    repo_root: Path,
    cache_root: Path,
) -> pd.DataFrame:
    clinical = build_clinical_feature_frame(cases, feature_set, repo_root=repo_root, cache_root=cache_root)
    clinical_columns = [column for column in clinical.columns if column not in CASE_ID_COLUMNS]
    if not clinical_columns:
        return features
    merged = features.merge(
        clinical,
        on=list(CASE_ID_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    return merged


def label_for_case(row: pd.Series, args: argparse.Namespace) -> float:
    if args.label_mode == "patient_progression":
        return float(row["clinical_progression"])

    if int(row["clinical_progression"]) == 0:
        return 0.0

    mri_day = row.get("days_from_diagnosis_to_mri")
    prog_day = row.get("clinical_number_of_days_from_diagnosis_to_date_of_first_progression")
    if pd.isna(prog_day):
        prog_day = row.get("clinical_time_to_first_progression_days")
    if pd.isna(mri_day):
        return np.nan

    if args.label_mode == "post_progression":
        if pd.isna(prog_day):
            return 0.0 if int(row["clinical_progression"]) == 0 else np.nan
        return float(float(mri_day) >= float(prog_day))

    if args.label_mode == "within_window":
        if pd.isna(prog_day):
            return 0.0 if int(row["clinical_progression"]) == 0 else np.nan
        delta = float(prog_day) - float(mri_day)
        return float(delta <= float(args.progression_window_days))

    raise ValueError(f"Unsupported label mode: {args.label_mode}")


def build_case_table(index_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    usable = index_df[index_df["roi_status"] == "written"].copy()
    usable = usable.sort_values(["patient_id", "timepoint_number"], kind="stable")
    for column in [*MODALITY_PATH_COLUMNS.values(), "multiclass_mask_path"]:
        usable = usable[usable[column].astype(str).ne("")]
    usable["union_voxels"] = (
        usable[["label1_voxels", "label2_voxels", "label3_voxels"]].fillna(0).sum(axis=1).astype(int)
    )
    usable = usable[usable["union_voxels"] > 0].copy()

    usable["label"] = usable.apply(lambda row: label_for_case(row, args), axis=1)
    usable["label"] = pd.to_numeric(usable["label"], errors="coerce")
    usable["progression_day"] = usable["clinical_number_of_days_from_diagnosis_to_date_of_first_progression"].fillna(
        usable["clinical_time_to_first_progression_days"]
    )
    usable["late_treatment_start_day"] = earliest_nonnegative_day(usable, LATE_TREATMENT_START_COLUMNS)
    usable["delta_to_progression_days"] = usable["progression_day"] - usable["days_from_diagnosis_to_mri"]
    usable = usable[usable["label"].isin([0.0, 1.0])].copy()
    usable["label"] = usable["label"].astype(int)
    if args.exclude_after_late_treatment:
        late_treatment_mask = usable["late_treatment_start_day"].isna() | (
            usable["days_from_diagnosis_to_mri"].notna()
            & (usable["days_from_diagnosis_to_mri"] < usable["late_treatment_start_day"])
        )
        usable = usable[late_treatment_mask].copy()
    if args.pre_progression_only:
        known_pre_progression = usable["progression_day"].notna() & usable["days_from_diagnosis_to_mri"].notna()
        pre_progression_mask = usable["clinical_progression"].eq(0) | (
            known_pre_progression & (usable["days_from_diagnosis_to_mri"] < usable["progression_day"])
        )
        usable = usable[pre_progression_mask].copy()
    if args.earliest_scan_only:
        usable = (
            usable.sort_values(["patient_id", "timepoint_number", "timepoint"], kind="stable")
            .groupby("patient_id", as_index=False, sort=False)
            .head(1)
            .copy()
        )
    if args.max_cases is not None:
        usable = usable.head(args.max_cases).copy()
    return usable


def union_mask_from_multiclass(mask_path: Path, output_path: Path) -> tuple[Path, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        mask_img = sitk.ReadImage(str(output_path), sitk.sitkUInt8)
        voxel_count = int((sitk.GetArrayFromImage(mask_img) > 0).sum())
        return output_path, voxel_count

    multiclass = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    array = sitk.GetArrayFromImage(multiclass)
    union = np.isin(array, UNION_LABELS).astype(np.uint8)
    union_img = sitk.GetImageFromArray(union)
    union_img.CopyInformation(multiclass)
    sitk.WriteImage(union_img, str(output_path))
    return output_path, int(union.sum())


def preprocess_image(input_path: Path, mask_path: Path, output_path: Path) -> tuple[Path, dict[str, float]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path, {}

    image = sitk.ReadImage(str(input_path), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(image, mask)

    data = sitk.GetArrayFromImage(corrected).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask) > 0
    voxels = data[mask_arr]
    if voxels.size == 0:
        raise ValueError(f"No positive voxels in union mask for {input_path}")
    mean_value = float(voxels.mean())
    std_value = float(voxels.std())
    if std_value <= EPS:
        std_value = 1.0
    normalized = (data - mean_value) / std_value
    normalized[~np.isfinite(normalized)] = 0.0
    out = sitk.GetImageFromArray(normalized.astype(np.float32, copy=False))
    out.CopyInformation(corrected)
    sitk.WriteImage(out, str(output_path))
    return output_path, {
        "mask_mean": mean_value,
        "mask_std": std_value,
        "mask_voxels": int(mask_arr.sum()),
    }


def scalarize_radiomics_value(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(()))
    return None


def build_radiomics_extractor(yaml_path: Path) -> featureextractor.RadiomicsFeatureExtractor:
    warnings.filterwarnings("ignore")
    radiomics_logger.setLevel(logging.ERROR)
    return featureextractor.RadiomicsFeatureExtractor(str(yaml_path))


def preprocess_and_extract_case(
    case: dict[str, object],
    repo_root_str: str,
    cache_root_str: str,
    yaml_path_str: str,
    reuse_case_features: bool,
) -> dict[str, object]:
    repo_root = Path(repo_root_str)
    cache_root = Path(cache_root_str)
    yaml_path = Path(yaml_path_str)
    patient_id = str(case["patient_id"])
    timepoint = str(case["timepoint"])
    case_dir = cache_root / "preprocessed" / patient_id / timepoint
    feature_path = cache_root / "features" / patient_id / f"{timepoint}.json"
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    if reuse_case_features and feature_path.exists():
        cached = json.loads(feature_path.read_text(encoding="utf-8"))
        metadata = case_metadata_record(
            case,
            union_voxels=int(cached.get("union_mask_voxels", case.get("union_voxels", 0))),
        )
        for key, value in cached.items():
            if key not in metadata:
                metadata[key] = value
        return metadata

    mask_source = resolve_repo_path(repo_root, case["multiclass_mask_path"])
    union_mask_path, union_voxels = union_mask_from_multiclass(
        mask_path=mask_source,
        output_path=case_dir / "tumor_mask_union.nii.gz",
    )
    if union_voxels <= 0:
        raise ValueError(f"Empty union mask for {patient_id} {timepoint}")

    extractor = build_radiomics_extractor(yaml_path)
    record: dict[str, object] = case_metadata_record(case, union_voxels=union_voxels)

    for modality, path_column in MODALITY_PATH_COLUMNS.items():
        input_path = resolve_repo_path(repo_root, case[path_column])
        output_path = case_dir / f"{modality}.nii.gz"
        _, stats = preprocess_image(input_path=input_path, mask_path=union_mask_path, output_path=output_path)
        for key, value in stats.items():
            record[f"{modality}_{key}"] = value
        result = extractor.execute(str(output_path), str(union_mask_path))
        for feature_name, value in result.items():
            if feature_name.startswith("diagnostics_"):
                continue
            numeric_value = scalarize_radiomics_value(value)
            if numeric_value is not None:
                record[f"{modality}_{feature_name.replace('original_', '')}"] = numeric_value

    feature_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    return record


def extract_feature_table(cases: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    repo_root = args.repo_root.resolve()
    cache_root = args.cache_root.resolve()
    yaml_path = args.radiomics_yaml.resolve()
    case_records = cases.to_dict(orient="records")
    rows: list[dict[str, object]] = []
    use_tqdm = progress_bar_enabled(args)

    with ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [
            executor.submit(
                preprocess_and_extract_case,
                case,
                str(repo_root),
                str(cache_root),
                str(yaml_path),
                args.reuse_case_features,
            )
            for case in case_records
        ]
        total = len(futures)
        progress = tqdm(total=total, desc="Radiomics", disable=not use_tqdm) if use_tqdm else None
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if progress is not None:
                progress.update(1)
            if index % 10 == 0 or index == total:
                print(f"Extracted surveillance radiomics for {index}/{total} cases", flush=True)
        if progress is not None:
            progress.close()

    features = pd.DataFrame(rows).sort_values(["patient_id", "timepoint_number", "timepoint"], kind="stable")
    return features


def choose_test_patients(cases: pd.DataFrame, args: argparse.Namespace) -> tuple[set[str], dict[str, float | int]]:
    patient_stats = (
        cases.groupby("patient_id")
        .agg(
            sample_count=("label", "size"),
            positive_count=("label", "sum"),
        )
        .sort_index()
    )
    patient_ids = patient_stats.index.to_numpy()
    sample_counts = patient_stats["sample_count"].to_numpy(dtype=int)
    positive_counts = patient_stats["positive_count"].to_numpy(dtype=int)
    negative_counts = sample_counts - positive_counts

    rng = np.random.default_rng(args.seed)
    best_choice: np.ndarray | None = None
    best_score: tuple[int, int, int] | None = None
    best_summary: dict[str, float | int] = {}

    for _ in range(args.split_search_iters):
        choice = rng.choice(len(patient_ids), size=args.target_test_patients, replace=False)
        sample_total = int(sample_counts[choice].sum())
        positive_total = int(positive_counts[choice].sum())
        negative_total = int(negative_counts[choice].sum())
        score = (
            abs(sample_total - args.target_test_samples),
            abs(positive_total - args.target_test_positives),
            abs(negative_total - (args.target_test_samples - args.target_test_positives)),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_choice = choice
            best_summary = {
                "target_patients": int(args.target_test_patients),
                "selected_patients": int(len(choice)),
                "target_samples": int(args.target_test_samples),
                "selected_samples": sample_total,
                "target_positives": int(args.target_test_positives),
                "selected_positives": positive_total,
                "target_negatives": int(args.target_test_samples - args.target_test_positives),
                "selected_negatives": negative_total,
                "sample_gap": int(score[0]),
                "positive_gap": int(score[1]),
                "negative_gap": int(score[2]),
            }
            if score == (0, 0, 0):
                break

    assert best_choice is not None
    return set(patient_ids[best_choice].tolist()), best_summary


def summarize_split(features: pd.DataFrame, test_patients: set[str], args: argparse.Namespace) -> dict[str, int]:
    patient_ids = set(features["patient_id"].astype(str).unique().tolist())
    unknown_patients = sorted(test_patients - patient_ids)
    if unknown_patients:
        raise ValueError(f"Unknown held-out patients: {unknown_patients[:10]}")

    test_mask = features["patient_id"].astype(str).isin(test_patients)
    split_counts = (
        pd.DataFrame({"split": np.where(test_mask, "test", "train"), "label": features["label"]})
        .groupby("split")["label"]
        .agg(["size", "sum"])
    )
    if "test" not in split_counts.index or "train" not in split_counts.index:
        raise ValueError("Both train and test splits must contain at least one sample.")

    test_samples = int(split_counts.loc["test", "size"])
    test_positives = int(split_counts.loc["test", "sum"])
    test_negatives = int(test_samples - test_positives)
    train_samples = int(split_counts.loc["train", "size"])
    train_positives = int(split_counts.loc["train", "sum"])
    train_negatives = int(train_samples - train_positives)
    return {
        "target_patients": int(args.target_test_patients),
        "selected_patients": int(len(test_patients)),
        "target_samples": int(args.target_test_samples),
        "selected_samples": test_samples,
        "target_positives": int(args.target_test_positives),
        "selected_positives": test_positives,
        "target_negatives": int(args.target_test_samples - args.target_test_positives),
        "selected_negatives": test_negatives,
        "sample_gap": int(abs(test_samples - args.target_test_samples)),
        "positive_gap": int(abs(test_positives - args.target_test_positives)),
        "negative_gap": int(abs(test_negatives - (args.target_test_samples - args.target_test_positives))),
        "train_samples": train_samples,
        "train_positives": train_positives,
        "train_negatives": train_negatives,
    }


def candidate_feature_columns(
    df: pd.DataFrame,
    allowed_modalities: list[str] | None = None,
    include_clinical: bool = False,
    include_engineered: bool = False,
) -> list[str]:
    metadata_cols = set(CASE_METADATA_COLUMNS)
    excluded_suffixes = ("_mask_mean", "_mask_std", "_mask_voxels")
    allowed = set(allowed_modalities or ALL_MODALITIES)
    return [
        column
        for column in df.columns
        if column not in metadata_cols
        and not column.endswith(excluded_suffixes)
        and (
            column.split("_", 1)[0] in allowed
            or (include_clinical and column.startswith(CLINICAL_FEATURE_PREFIX))
            or (include_engineered and column.startswith(ENGINEERED_FEATURE_PREFIX))
        )
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def correlation_keep_columns(X: pd.DataFrame, threshold: float) -> list[str]:
    if X.shape[1] <= 1:
        return X.columns.tolist()
    corr = X.corr(method="spearman").abs()
    values = corr.to_numpy()
    np.fill_diagonal(values, 0.0)
    high_pairs = np.argwhere(np.triu(values >= threshold, k=1))
    if high_pairs.size == 0:
        return X.columns.tolist()

    adjacency = [set() for _ in range(values.shape[0])]
    for left, right in high_pairs.tolist():
        adjacency[left].add(right)
        adjacency[right].add(left)

    mean_corr = values.mean(axis=0)
    order = np.argsort(-mean_corr, kind="stable")
    keep = np.ones(values.shape[0], dtype=bool)
    for index in order:
        if not keep[index]:
            continue
        if any(keep[neighbor] for neighbor in adjacency[index]):
            keep[index] = False

    kept = [column for column, flag in zip(X.columns.tolist(), keep.tolist()) if flag]
    if not kept:
        kept = [X.columns[int(np.argmax(values.mean(axis=0)))]]
    return kept


def fit_preprocessor(
    X: pd.DataFrame,
    variance_threshold: float,
    corr_threshold: float,
    scale: bool,
) -> tuple[pd.DataFrame, PreprocessorState]:
    medians = X.median(axis=0).fillna(0.0)
    imputed = X.fillna(medians)
    variances = imputed.var(axis=0, ddof=0)
    keep_variance = variances[variances > variance_threshold].index.tolist()
    if not keep_variance:
        keep_variance = imputed.columns.tolist()
    reduced = imputed[keep_variance]
    keep_columns = correlation_keep_columns(reduced, corr_threshold)
    reduced = reduced[keep_columns]

    scale_mean = None
    scale_scale = None
    if scale:
        scaler = StandardScaler()
        reduced = pd.DataFrame(
            scaler.fit_transform(reduced),
            index=reduced.index,
            columns=reduced.columns,
        )
        scale_mean = scaler.mean_
        scale_scale = scaler.scale_

    state = PreprocessorState(
        medians=medians,
        keep_columns=keep_columns,
        scale_mean=scale_mean,
        scale_scale=scale_scale,
        scale=scale,
        variance_threshold=variance_threshold,
        corr_threshold=corr_threshold,
    )
    return reduced, state


def transform_preprocessor(X: pd.DataFrame, state: PreprocessorState) -> pd.DataFrame:
    imputed = X.reindex(columns=state.medians.index).fillna(state.medians)
    reduced = imputed[state.keep_columns].copy()
    if state.scale and state.scale_mean is not None and state.scale_scale is not None:
        denom = np.where(np.asarray(state.scale_scale) <= EPS, 1.0, np.asarray(state.scale_scale))
        reduced = pd.DataFrame(
            (reduced.to_numpy(dtype=float) - np.asarray(state.scale_mean)) / denom,
            index=reduced.index,
            columns=reduced.columns,
        )
    return reduced


def cv_splitter(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int, group_aware: bool) -> object:
    if group_aware:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def build_model(model_name: str, params: dict[str, object], seed: int, lightgbm_device: str):
    if model_name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            metric="auc",
            verbosity=-1,
            is_unbalance=True,
            deterministic=True,
            force_row_wise=True,
            device_type=lightgbm_device,
            random_state=seed,
            n_jobs=1,
            **params,
        )
    if model_name == "logreg":
        return LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=10000,
            class_weight="balanced",
            random_state=seed,
            **params,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
            **params,
        )
    if model_name == "svm":
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=seed,
            **params,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    decision = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-decision))


def trial_params(trial: optuna.trial.Trial, model_name: str) -> tuple[dict[str, object], float, float]:
    variance_threshold = float(trial.suggest_float("variance_threshold", 1e-12, 1e-6, log=True))
    corr_threshold = float(trial.suggest_float("corr_threshold", 0.5, 0.95))

    if model_name == "lightgbm":
        return {
            "learning_rate": float(trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)),
            "n_estimators": int(trial.suggest_int("n_estimators", 50, 1000)),
            "num_leaves": int(trial.suggest_int("num_leaves", 8, 256)),
            "min_child_samples": int(trial.suggest_int("min_child_samples", 5, 100)),
            "subsample": float(trial.suggest_float("subsample", 0.5, 1.0)),
            "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.3, 1.0)),
            "reg_alpha": float(trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)),
            "reg_lambda": float(trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)),
        }, variance_threshold, corr_threshold
    if model_name == "logreg":
        return {
            "C": float(trial.suggest_float("C", 1e-3, 10.0, log=True)),
        }, variance_threshold, corr_threshold
    if model_name == "rf":
        return {
            "n_estimators": int(trial.suggest_int("n_estimators", 100, 1000)),
            "max_depth": int(trial.suggest_int("max_depth", 2, 20)),
            "min_samples_leaf": int(trial.suggest_int("min_samples_leaf", 1, 20)),
            "max_features": float(trial.suggest_float("max_features", 0.2, 1.0)),
        }, variance_threshold, corr_threshold
    if model_name == "svm":
        return {
            "C": float(trial.suggest_float("C", 1e-3, 100.0, log=True)),
            "gamma": float(trial.suggest_float("gamma", 1e-4, 1.0, log=True)),
        }, variance_threshold, corr_threshold
    raise ValueError(model_name)


def model_uses_scaling(model_name: str) -> bool:
    return model_name in {"lightgbm", "logreg", "svm"}


def cross_validated_auc(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_name: str,
    params: dict[str, object],
    variance_threshold: float,
    corr_threshold: float,
    n_splits: int,
    seed: int,
    lightgbm_device: str,
) -> dict[str, float]:
    splitter = cv_splitter(y.to_numpy(), groups.to_numpy(), n_splits=n_splits, seed=seed, group_aware=True)
    fold_aucs: list[float] = []
    for train_idx, val_idx in splitter.split(X, y, groups):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        if y_train.nunique() < 2:
            continue
        X_train_proc, state = fit_preprocessor(
            X_train,
            variance_threshold=variance_threshold,
            corr_threshold=corr_threshold,
            scale=model_uses_scaling(model_name),
        )
        X_val_proc = transform_preprocessor(X_val, state)
        model = build_model(model_name, params, seed, lightgbm_device)
        model.fit(X_train_proc, y_train)
        prob = predict_positive_probability(model, X_val_proc)
        if np.unique(y_val).size == 2:
            fold_aucs.append(float(roc_auc_score(y_val, prob)))

    if not fold_aucs:
        raise ValueError("No valid fold AUCs were produced.")
    return {
        "best_fold_auc": float(max(fold_aucs)),
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "min_fold_auc": float(min(fold_aucs)),
    }


def rank_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[str]]:
    filtered_train, filtered_state = fit_preprocessor(
        X_train,
        variance_threshold=1e-8,
        corr_threshold=0.80,
        scale=False,
    )
    filtered_columns = filtered_train.columns.tolist()
    splitter = cv_splitter(
        y_train.to_numpy(),
        groups_train.to_numpy(),
        n_splits=min(args.ranking_folds, max(2, y_train.value_counts().min())),
        seed=args.seed,
        group_aware=True,
    )

    coefficient_series: list[pd.Series] = []
    permutation_series: list[pd.Series] = []
    X_rank = transform_preprocessor(X_train, filtered_state)

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X_rank, y_train, groups_train), start=1):
        X_fold_train = X_rank.iloc[train_idx]
        X_fold_val = X_rank.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        if y_fold_train.nunique() < 2 or y_fold_val.nunique() < 2:
            continue

        scaler = StandardScaler()
        X_fold_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_fold_train),
            index=X_fold_train.index,
            columns=X_fold_train.columns,
        )
        X_fold_val_scaled = pd.DataFrame(
            scaler.transform(X_fold_val),
            index=X_fold_val.index,
            columns=X_fold_val.columns,
        )
        model = LogisticRegression(
            solver="saga",
            penalty="l1",
            C=1.0,
            max_iter=10000,
            class_weight="balanced",
            random_state=args.seed,
        )
        model.fit(X_fold_train_scaled, y_fold_train)
        coefficient_series.append(
            pd.Series(np.abs(model.coef_.ravel()), index=X_fold_train_scaled.columns, name=f"coef_fold_{fold_index}")
        )
        perm = permutation_importance(
            model,
            X_fold_val_scaled,
            y_fold_val,
            scoring="roc_auc",
            n_repeats=args.permutation_repeats,
            random_state=args.seed,
            n_jobs=1,
        )
        permutation_series.append(
            pd.Series(perm.importances_mean, index=X_fold_val_scaled.columns, name=f"perm_fold_{fold_index}")
        )

    if not coefficient_series or not permutation_series:
        raise ValueError("No valid ranking folds were available after class checks.")

    coef_df = pd.concat(coefficient_series, axis=1).fillna(0.0)
    perm_df = pd.concat(permutation_series, axis=1).fillna(0.0)
    ranking = pd.DataFrame(
        {
            "feature": filtered_columns,
            "median_abs_l1": coef_df.median(axis=1).reindex(filtered_columns).fillna(0.0).to_numpy(),
            "median_permutation_importance": perm_df.median(axis=1)
            .reindex(filtered_columns)
            .fillna(0.0)
            .to_numpy(),
        }
    )
    ranking["l1_rank"] = ranking["median_abs_l1"].rank(ascending=False, method="average")
    ranking["perm_rank"] = ranking["median_permutation_importance"].rank(ascending=False, method="average")
    ranking["consensus_rank"] = ranking["l1_rank"] + ranking["perm_rank"]
    ranking = ranking.sort_values(
        ["consensus_rank", "median_permutation_importance", "median_abs_l1", "feature"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return ranking, filtered_columns


def optimize_models(
    train_df: pd.DataFrame,
    ranking: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_cols = ranking["feature"].tolist()
    subset_sizes = [
        min(int(value), len(feature_cols))
        for value in args.feature_subsets.split(",")
        if str(value).strip()
    ]
    subset_sizes = sorted(set(size for size in subset_sizes if size > 0), reverse=True)
    model_names = [value.strip() for value in args.models.split(",") if value.strip()]

    X_train_full = train_df[feature_cols]
    y_train = train_df["label"]
    groups_train = train_df["patient_id"]
    search_rows: list[dict[str, object]] = []
    best_choice: dict[str, object] | None = None
    total_combinations = max(1, len(subset_sizes) * len(model_names))
    completed_combinations = 0
    use_tqdm = progress_bar_enabled(args)
    combination_bar = tqdm(total=total_combinations, desc="Model configs", disable=not use_tqdm) if use_tqdm else None

    try:
        for subset_size in subset_sizes:
            chosen_columns = feature_cols[:subset_size]
            X_subset = X_train_full[chosen_columns]
            for model_name in model_names:
                combination_index = completed_combinations + 1
                print(
                    f"Starting model search {combination_index}/{total_combinations}: {model_name} top-{subset_size}",
                    flush=True,
                )
                write_progress(
                    args.output_dir,
                    {
                        "stage": "model_search",
                        "current_model": model_name,
                        "current_subset_size": int(subset_size),
                        "current_combination": int(combination_index),
                        "total_combinations": int(total_combinations),
                        "completed_combinations": int(completed_combinations),
                        "completed_trials": 0,
                        "total_trials": int(args.n_trials),
                    },
                )
                sampler = optuna.samplers.TPESampler(seed=args.seed)
                study = optuna.create_study(direction="maximize", sampler=sampler)
                trial_bar = (
                    tqdm(
                        total=int(args.n_trials),
                        desc=f"{model_name} top-{subset_size}",
                        leave=False,
                        disable=not use_tqdm,
                    )
                    if use_tqdm
                    else None
                )

                def objective(trial: optuna.trial.Trial) -> float:
                    params, variance_threshold, corr_threshold = trial_params(trial, model_name)
                    metrics = cross_validated_auc(
                        X=X_subset,
                        y=y_train,
                        groups=groups_train,
                        model_name=model_name,
                        params=params,
                        variance_threshold=variance_threshold,
                        corr_threshold=corr_threshold,
                        n_splits=args.cv_folds,
                        seed=args.seed,
                        lightgbm_device=args.lightgbm_device,
                    )
                    trial.set_user_attr("best_fold_auc", metrics["best_fold_auc"])
                    trial.set_user_attr("mean_fold_auc", metrics["mean_fold_auc"])
                    trial.set_user_attr("min_fold_auc", metrics["min_fold_auc"])
                    return metrics["mean_fold_auc"]

                def progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
                    completed_trials = int(trial.number + 1)
                    best_value = float(study.best_value) if study.best_trial is not None else None
                    write_progress(
                        args.output_dir,
                        {
                            "stage": "model_search",
                            "current_model": model_name,
                            "current_subset_size": int(subset_size),
                            "current_combination": int(combination_index),
                            "total_combinations": int(total_combinations),
                            "completed_combinations": int(completed_combinations),
                            "completed_trials": completed_trials,
                            "total_trials": int(args.n_trials),
                            "best_mean_fold_auc_so_far": best_value,
                        },
                    )
                    if trial_bar is not None:
                        trial_bar.update(completed_trials - trial_bar.n)
                        if best_value is not None:
                            trial_bar.set_postfix(best_mean_auc=f"{best_value:.4f}")
                    if completed_trials == 1 or completed_trials % 10 == 0 or completed_trials == int(args.n_trials):
                        best_text = "nan" if best_value is None else f"{best_value:.4f}"
                        print(
                            f"Search {combination_index}/{total_combinations}: {model_name} top-{subset_size} "
                            f"trial {completed_trials}/{args.n_trials} best_mean_auc={best_text}",
                            flush=True,
                        )

                study.optimize(
                    objective,
                    n_trials=args.n_trials,
                    n_jobs=1,
                    show_progress_bar=False,
                    callbacks=[progress_callback],
                )
                if trial_bar is not None:
                    trial_bar.close()
                best_trial = study.best_trial
                row = {
                    "model": model_name,
                    "subset_size": subset_size,
                    "best_fold_auc": float(best_trial.user_attrs["best_fold_auc"]),
                    "mean_fold_auc": float(best_trial.user_attrs["mean_fold_auc"]),
                    "min_fold_auc": float(best_trial.user_attrs["min_fold_auc"]),
                    "best_params": json.dumps(best_trial.params, sort_keys=True),
                }
                search_rows.append(row)
                candidate = {
                    "model": model_name,
                    "subset_size": subset_size,
                    "best_params": dict(best_trial.params),
                    "columns": chosen_columns,
                    "objective": float(best_trial.value),
                    "best_fold_auc": float(best_trial.user_attrs["best_fold_auc"]),
                    "mean_fold_auc": float(best_trial.user_attrs["mean_fold_auc"]),
                    "min_fold_auc": float(best_trial.user_attrs["min_fold_auc"]),
                }
                key = (
                    candidate["objective"],
                    candidate["min_fold_auc"],
                    candidate["best_fold_auc"],
                    -subset_size,
                )
                best_key = None if best_choice is None else (
                    best_choice["objective"],
                    best_choice["min_fold_auc"],
                    best_choice["best_fold_auc"],
                    -best_choice["subset_size"],
                )
                if best_key is None or key > best_key:
                    best_choice = candidate
                completed_combinations += 1
                pd.DataFrame(search_rows).sort_values(
                    ["mean_fold_auc", "min_fold_auc", "best_fold_auc", "subset_size"],
                    ascending=[False, False, False, False],
                    kind="stable",
                ).to_csv(args.output_dir / "model_search.csv", index=False)
                write_progress(
                    args.output_dir,
                    {
                        "stage": "model_search",
                        "current_model": model_name,
                        "current_subset_size": int(subset_size),
                        "current_combination": int(combination_index),
                        "total_combinations": int(total_combinations),
                        "completed_combinations": int(completed_combinations),
                        "completed_trials": int(args.n_trials),
                        "total_trials": int(args.n_trials),
                        "last_completed_mean_fold_auc": float(best_trial.user_attrs["mean_fold_auc"]),
                        "best_overall_mean_fold_auc": float(best_choice["objective"]),
                        "best_overall_model": str(best_choice["model"]),
                        "best_overall_subset_size": int(best_choice["subset_size"]),
                    },
                )
                print(
                    f"Completed model search {combination_index}/{total_combinations}: {model_name} top-{subset_size} "
                    f"mean_auc={best_trial.user_attrs['mean_fold_auc']:.4f}",
                    flush=True,
                )
                if combination_bar is not None:
                    combination_bar.update(1)
    finally:
        if combination_bar is not None:
            combination_bar.close()

    assert best_choice is not None
    return pd.DataFrame(search_rows).sort_values(
        ["mean_fold_auc", "min_fold_auc", "best_fold_auc", "subset_size"],
        ascending=[False, False, False, False],
        kind="stable",
    ), best_choice


def fit_oof_predictions(
    train_df: pd.DataFrame,
    model_name: str,
    params: dict[str, object],
    columns: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, float]], PreprocessorState, object]:
    X_train = train_df[columns]
    y_train = train_df["label"]
    groups = train_df["patient_id"]
    variance_threshold = float(params["variance_threshold"])
    corr_threshold = float(params["corr_threshold"])
    model_params = {key: value for key, value in params.items() if key not in {"variance_threshold", "corr_threshold"}}

    splitter = cv_splitter(y_train.to_numpy(), groups.to_numpy(), args.cv_folds, args.seed, group_aware=True)
    oof = np.full(len(train_df), np.nan, dtype=float)
    fold_rows: list[dict[str, float]] = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train, groups), start=1):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]
        if y_fold_train.nunique() < 2:
            fallback = float(y_fold_train.iloc[0])
            oof[val_idx] = fallback
            fold_rows.append(
                {
                    "fold": fold_index,
                    "auc": float("nan"),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                }
            )
            continue
        X_fold_train_proc, state = fit_preprocessor(
            X_fold_train,
            variance_threshold=variance_threshold,
            corr_threshold=corr_threshold,
            scale=model_uses_scaling(model_name),
        )
        X_fold_val_proc = transform_preprocessor(X_fold_val, state)
        model = build_model(model_name, model_params, args.seed, args.lightgbm_device)
        model.fit(X_fold_train_proc, y_fold_train)
        prob = predict_positive_probability(model, X_fold_val_proc)
        oof[val_idx] = prob
        fold_rows.append(
            {
                "fold": fold_index,
                "auc": float(roc_auc_score(y_fold_val, prob)) if y_fold_val.nunique() == 2 else float("nan"),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
            }
        )

    final_train_proc, final_state = fit_preprocessor(
        X_train,
        variance_threshold=variance_threshold,
        corr_threshold=corr_threshold,
        scale=model_uses_scaling(model_name),
    )
    final_model = build_model(model_name, model_params, args.seed, args.lightgbm_device)
    final_model.fit(final_train_proc, y_train)
    return oof, fold_rows, final_state, final_model


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, int | float]]:
    thresholds = np.unique(np.quantile(probabilities, np.linspace(0.01, 0.99, 199)))
    best_threshold = 0.5
    best_stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "errors": len(y_true)}
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        errors = fp + fn
        if errors < best_stats["errors"]:
            best_threshold = float(threshold)
            best_stats = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "errors": errors}
    return best_threshold, best_stats


def auc_confidence_interval(y_true: np.ndarray, probabilities: np.ndarray, seed: int, n_bootstraps: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstraps):
        sample_idx = rng.integers(0, len(y_true), size=len(y_true))
        y_sample = y_true[sample_idx]
        if np.unique(y_sample).size < 2:
            continue
        values.append(float(roc_auc_score(y_sample, probabilities[sample_idx])))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision_class0": float(precision[0]),
        "recall_class0": float(recall[0]),
        "f1_class0": float(f1[0]),
        "precision_class1": float(precision[1]),
        "recall_class1": float(recall[1]),
        "f1_class1": float(f1[1]),
        "tp": int(((pred == 1) & (y_true == 1)).sum()),
        "tn": int(((pred == 0) & (y_true == 0)).sum()),
        "fp": int(((pred == 1) & (y_true == 0)).sum()),
        "fn": int(((pred == 0) & (y_true == 1)).sum()),
    }


def decision_curve(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    n = float(len(y_true))
    prevalence = float((y_true == 1).mean())
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        weight = threshold / (1.0 - threshold)
        nb_model = tp / n - fp / n * weight
        nb_all = prevalence - (1.0 - prevalence) * weight
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit_model": float(nb_model),
                "net_benefit_all": float(nb_all),
                "net_benefit_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_roc(output_path: Path, y_true: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> None:
    plt.figure(figsize=(6.0, 5.0))
    for label, values in (("Raw", raw), ("Calibrated", calibrated)):
        fpr, tpr, _ = roc_curve(y_true, values)
        auc_value = roc_auc_score(y_true, values)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc_value:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_decision_curve(output_path: Path, dca_df: pd.DataFrame) -> None:
    plt.figure(figsize=(6.4, 5.0))
    plt.plot(dca_df["threshold"], dca_df["net_benefit_model"], label="Model", linewidth=2)
    plt.plot(dca_df["threshold"], dca_df["net_benefit_all"], label="Treat-all", linestyle="--")
    plt.plot(dca_df["threshold"], dca_df["net_benefit_none"], label="Treat-none", linestyle="-.")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_shap_summary(
    model_name: str,
    fitted_model,
    final_state: PreprocessorState,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if model_name != "lightgbm":
        return pd.DataFrame(), {}
    transformed_test = transform_preprocessor(test_df[columns], final_state)
    explainer = shap.TreeExplainer(fitted_model)
    shap_values = explainer.shap_values(transformed_test)
    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[-1])
    else:
        shap_array = np.asarray(shap_values)
    mean_abs = np.abs(shap_array).mean(axis=0)
    summary = pd.DataFrame(
        {
            "feature": transformed_test.columns,
            "mean_abs_shap": mean_abs,
            "modality": [feature.split("_", 1)[0] for feature in transformed_test.columns],
        }
    ).sort_values("mean_abs_shap", ascending=False, kind="stable")
    modality_totals = summary.groupby("modality")["mean_abs_shap"].sum()
    modality_share = (modality_totals / modality_totals.sum()).to_dict()
    return summary, {key: float(value) for key, value in modality_share.items()}


def write_markdown_summary(output_path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Postoperative Progression Surveillance",
        "",
        "## Cohort",
        f"- Label mode: `{summary['label_mode']}`",
        f"- Modalities: `{','.join(summary['modalities'])}`",
        f"- Clinical feature set: `{summary['clinical_feature_set']}`",
        f"- Pre-progression only: `{summary['pre_progression_only']}`",
        f"- Earliest scan only: `{summary['earliest_scan_only']}`",
        f"- Exclude after late treatment: `{summary['exclude_after_late_treatment']}`",
        f"- Labeled cases: `{summary['cohort']['cases']}` across `{summary['cohort']['patients']}` patients",
        f"- Positive / negative: `{summary['cohort']['positives']}` / `{summary['cohort']['negatives']}`",
        f"- Held-out patients: `{summary['split_summary']['selected_patients']}`",
        f"- Held-out samples: `{summary['split_summary']['selected_samples']}`",
        f"- Held-out positives / negatives: `{summary['split_summary']['selected_positives']}` / `{summary['split_summary']['selected_negatives']}`",
        "",
        "## Model",
        f"- Best model: `{summary['best_model']['model']}`",
        f"- Feature subset size: `{summary['best_model']['subset_size']}`",
        f"- CV objective (mean fold AUC): `{summary['best_model']['objective']:.4f}`",
        f"- Worst fold AUC: `{summary['best_model']['min_fold_auc']:.4f}`",
        "",
        "## Test Metrics",
        f"- Raw ROC AUC: `{summary['test_metrics_raw']['roc_auc']:.4f}`",
        f"- Calibrated ROC AUC: `{summary['test_metrics_calibrated']['roc_auc']:.4f}`",
        f"- Raw Brier: `{summary['test_brier_raw']:.4f}`",
        f"- Calibrated Brier: `{summary['test_brier_calibrated']:.4f}`",
        f"- Threshold: `{summary['threshold']:.4f}`",
        "- Primary per-case output: `progression_risk_probability` in `test_predictions.csv`",
        f"- Confusion matrix (TN / FP / FN / TP): `{summary['test_metrics_calibrated']['tn']}` / `{summary['test_metrics_calibrated']['fp']}` / `{summary['test_metrics_calibrated']['fn']}` / `{summary['test_metrics_calibrated']['tp']}`",
    ]
    if summary.get("auc_ci_low") is not None and summary.get("auc_ci_high") is not None:
        lines.append(
            f"- Calibrated ROC AUC 95% bootstrap CI: `[{summary['auc_ci_low']:.4f}, {summary['auc_ci_high']:.4f}]`"
        )
    write_text(output_path, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")
    radiomics_logger.setLevel(logging.ERROR)
    args.modalities = parse_modalities(args.modalities)
    validate_args(args)

    args.repo_root = args.repo_root.resolve()
    args.cache_root = args.cache_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.feature_table is not None:
        args.feature_table = args.feature_table.resolve()
    if args.test_patients_file is not None:
        args.test_patients_file = args.test_patients_file.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(args.experiment_index)
    cases = build_case_table(index_df, args)
    cohort_summary = {
        "cases": int(len(cases)),
        "patients": int(cases["patient_id"].nunique()),
        "positives": int(cases["label"].sum()),
        "negatives": int((1 - cases["label"]).sum()),
    }
    write_json(args.output_dir / "cohort_summary.json", cohort_summary)
    write_progress(
        args.output_dir,
        {
            "stage": "cohort_ready",
            "cases": int(cohort_summary["cases"]),
            "patients": int(cohort_summary["patients"]),
            "positives": int(cohort_summary["positives"]),
            "negatives": int(cohort_summary["negatives"]),
        },
    )

    feature_csv = args.output_dir / "radiomics_features.csv"
    if args.feature_table is not None:
        reusable = pd.read_csv(args.feature_table)
        metadata = case_metadata_frame(cases)
        predictor_columns = [
            column
            for column in reusable.columns
            if column not in set(CASE_METADATA_COLUMNS)
            and not column.startswith(CLINICAL_FEATURE_PREFIX)
        ]
        feature_block = reusable[list(CASE_ID_COLUMNS) + predictor_columns].drop_duplicates(
            subset=list(CASE_ID_COLUMNS),
            keep="last",
        )
        features = metadata.merge(
            feature_block,
            on=list(CASE_ID_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        if len(features) != len(metadata):
            missing = metadata.merge(
                feature_block[list(CASE_ID_COLUMNS)],
                on=list(CASE_ID_COLUMNS),
                how="left",
                indicator=True,
            )
            missing_rows = missing[missing["_merge"] != "both"]
            raise ValueError(
                f"Feature table {args.feature_table} is missing {len(missing_rows)} current cohort rows."
            )
    else:
        features = extract_feature_table(cases, args)
    features = merge_clinical_features(
        features,
        cases,
        args.clinical_feature_set,
        repo_root=args.repo_root,
        cache_root=args.cache_root,
    )
    features.to_csv(feature_csv, index=False)
    write_progress(
        args.output_dir,
        {
            "stage": "feature_table_ready",
            "feature_table": feature_csv.name,
            "rows": int(len(features)),
            "columns": int(len(features.columns)),
        },
    )

    if args.test_patients_file is not None:
        test_patients = {
            line.strip()
            for line in args.test_patients_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    else:
        test_patients, _ = choose_test_patients(features, args)
    split_summary = summarize_split(features, test_patients, args)
    features["split"] = np.where(features["patient_id"].isin(test_patients), "test", "train")
    write_text(
        args.output_dir / "test_patients.txt",
        "\n".join(sorted(test_patients)) + "\n",
    )
    write_text(
        args.output_dir / "train_patients.txt",
        "\n".join(sorted(set(features["patient_id"]) - set(test_patients))) + "\n",
    )
    write_json(args.output_dir / "split_summary.json", split_summary)
    write_progress(
        args.output_dir,
        {
            "stage": "split_ready",
            **{key: int(value) for key, value in split_summary.items()},
        },
    )

    train_df = features[features["split"] == "train"].copy()
    test_df = features[features["split"] == "test"].copy()
    all_feature_columns = candidate_feature_columns(
        features,
        allowed_modalities=args.modalities,
        include_clinical=args.clinical_feature_set != "none",
        include_engineered=args.clinical_feature_set
        in {"hybrid_engineered", "hybrid_engineered_biologic", "report_core", "report_timing", "report_full"},
    )
    if not all_feature_columns:
        raise ValueError(f"No feature columns found for selected modalities: {args.modalities}")
    ranking, filtered_columns = rank_features(train_df[all_feature_columns], train_df["label"], train_df["patient_id"], args)
    ranking.to_csv(args.output_dir / "feature_ranking.csv", index=False)
    write_progress(
        args.output_dir,
        {
            "stage": "ranking_complete",
            "pre_filter_feature_count": int(len(all_feature_columns)),
            "post_filter_feature_count": int(len(filtered_columns)),
        },
    )

    search_df, best_choice = optimize_models(train_df, ranking, args)
    search_df.to_csv(args.output_dir / "model_search.csv", index=False)
    write_progress(
        args.output_dir,
        {
            "stage": "model_search_complete",
            "best_model": str(best_choice["model"]),
            "best_subset_size": int(best_choice["subset_size"]),
            "best_mean_fold_auc": float(best_choice["objective"]),
            "best_min_fold_auc": float(best_choice["min_fold_auc"]),
        },
    )
    best_params = dict(best_choice["best_params"])
    best_params["variance_threshold"] = float(best_params["variance_threshold"])
    best_params["corr_threshold"] = float(best_params["corr_threshold"])

    oof_raw, fold_rows, final_state, final_model = fit_oof_predictions(
        train_df=train_df,
        model_name=str(best_choice["model"]),
        params=best_params,
        columns=list(best_choice["columns"]),
        args=args,
    )
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "cv_fold_metrics.csv", index=False)
    write_progress(
        args.output_dir,
        {
            "stage": "calibration_evaluation",
            "best_model": str(best_choice["model"]),
            "best_subset_size": int(best_choice["subset_size"]),
            "train_oof_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
    )

    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=args.seed)
    calibrator.fit(oof_raw.reshape(-1, 1), train_df["label"])
    oof_cal = calibrator.predict_proba(oof_raw.reshape(-1, 1))[:, 1]
    threshold, threshold_stats = select_threshold(train_df["label"].to_numpy(), oof_cal)

    transformed_test = transform_preprocessor(test_df[list(best_choice["columns"])], final_state)
    test_raw = predict_positive_probability(final_model, transformed_test)
    test_cal = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
    test_metrics_raw = threshold_metrics(test_df["label"].to_numpy(), test_raw, threshold)
    test_metrics_cal = threshold_metrics(test_df["label"].to_numpy(), test_cal, threshold)
    brier_raw = float(brier_score_loss(test_df["label"], test_raw))
    brier_cal = float(brier_score_loss(test_df["label"], test_cal))
    auc_ci_low, auc_ci_high = auc_confidence_interval(
        test_df["label"].to_numpy(),
        test_cal,
        seed=args.seed,
        n_bootstraps=args.bootstrap_iterations,
    )

    predictions = test_df[
        [
            "patient_id",
            "timepoint",
            "timepoint_number",
            "label",
            "days_from_diagnosis_to_mri",
            "progression_day",
            "delta_to_progression_days",
        ]
    ].copy()
    predictions["raw_probability"] = test_raw
    predictions["calibrated_probability"] = test_cal
    predictions["progression_risk_probability"] = test_cal
    predictions["progression_risk_percent"] = test_cal * 100.0
    predictions["predicted_class"] = (test_cal >= threshold).astype(int)
    predictions["predicted_class_by_threshold"] = predictions["predicted_class"]
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)

    dca = decision_curve(test_df["label"].to_numpy(), test_cal)
    dca.to_csv(args.output_dir / "decision_curve.csv", index=False)
    plot_roc(args.output_dir / "roc_curve.png", test_df["label"].to_numpy(), test_raw, test_cal)
    plot_decision_curve(args.output_dir / "decision_curve.png", dca)

    shap_summary, modality_share = compute_shap_summary(
        model_name=str(best_choice["model"]),
        fitted_model=final_model,
        final_state=final_state,
        train_df=train_df,
        test_df=test_df,
        columns=list(best_choice["columns"]),
    )
    if not shap_summary.empty:
        shap_summary.to_csv(args.output_dir / "shap_feature_importance.csv", index=False)

    summary = {
        "label_mode": args.label_mode,
        "modalities": list(args.modalities),
        "clinical_feature_set": str(args.clinical_feature_set),
        "pre_progression_only": bool(args.pre_progression_only),
        "earliest_scan_only": bool(args.earliest_scan_only),
        "exclude_after_late_treatment": bool(args.exclude_after_late_treatment),
        "progression_window_days": int(args.progression_window_days),
        "cohort": cohort_summary,
        "split_summary": split_summary,
        "pre_filter_feature_count": int(len(all_feature_columns)),
        "post_filter_feature_count": int(len(filtered_columns)),
        "best_model": {
            "model": str(best_choice["model"]),
            "subset_size": int(best_choice["subset_size"]),
            "objective": float(best_choice["objective"]),
            "best_fold_auc": float(best_choice["best_fold_auc"]),
            "mean_fold_auc": float(best_choice["mean_fold_auc"]),
            "min_fold_auc": float(best_choice["min_fold_auc"]),
            "columns": list(best_choice["columns"]),
            "params": best_params,
        },
        "threshold": float(threshold),
        "threshold_stats": threshold_stats,
        "train_oof_auc_raw": float(roc_auc_score(train_df["label"], oof_raw)),
        "train_oof_auc_calibrated": float(roc_auc_score(train_df["label"], oof_cal)),
        "train_brier_raw": float(brier_score_loss(train_df["label"], oof_raw)),
        "train_brier_calibrated": float(brier_score_loss(train_df["label"], oof_cal)),
        "test_metrics_raw": test_metrics_raw,
        "test_metrics_calibrated": test_metrics_cal,
        "test_brier_raw": brier_raw,
        "test_brier_calibrated": brier_cal,
        "auc_ci_low": auc_ci_low,
        "auc_ci_high": auc_ci_high,
        "shap_modality_share": modality_share,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_markdown_summary(args.output_dir / "summary.md", summary)
    write_progress(
        args.output_dir,
        {
            "stage": "completed",
            "best_model": str(best_choice["model"]),
            "best_subset_size": int(best_choice["subset_size"]),
            "test_roc_auc": float(summary["test_metrics_calibrated"]["roc_auc"]),
            "test_brier_calibrated": float(summary["test_brier_calibrated"]),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
