#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


MOLECULAR_FEATURE_COLUMNS = {
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
}

HYBRID_BASIC_EXTRA_COLUMNS = {
    "clinical_age_at_diagnosis",
    "clinical_sex_at_birth",
}

ENGINEERED_CATEGORICAL_COLUMNS = {
    "clinical_grade_of_primary_brain_tumor",
    "clinical_primary_diagnosis",
    "clinical_previous_brain_tumor",
    "clinical_grade_of_previous_brain_tumor",
}

LEAKY_NAME_PATTERNS = (
    r"progression",
    r"hospice",
    r"death",
    r"survival",
    r"additional_therapy",
    r"2nd_additional_therapy",
    r"immuno",
    r"new_treatment",
    r"brachy",
    r"complete_",
    r"end_date",
    r"further_progression",
)

TIMING_MAYBE_SAFE_PATTERNS = (
    r"first_surgery_or_procedure",
    r"radiation_therapy_start",
    r"initial_chemo_therapy_start",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit clinical columns for completeness and leakage risk.")
    parser.add_argument(
        "--experiment-index",
        type=Path,
        default=Path("processed/manifests/experiment_index.csv"),
        help="Merged experiment index with clinical labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/clinical_feature_audit"),
        help="Directory for audit outputs.",
    )
    parser.add_argument(
        "--forward-cohort-only",
        action="store_true",
        help="Restrict the audit to the forward cohort used by the main surveillance model.",
    )
    return parser.parse_args()


def earliest_nonnegative_day(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index, dtype=float)
    values = df[available].apply(pd.to_numeric, errors="coerce")
    values = values.where(values >= 0)
    return values.min(axis=1, skipna=True)


def build_forward_cohort(index_df: pd.DataFrame) -> pd.DataFrame:
    usable = index_df[index_df["roi_status"] == "written"].copy()
    usable = usable.sort_values(["patient_id", "timepoint_number"], kind="stable")
    path_columns = [
        "native_t1_path",
        "native_t1c_path",
        "native_flair_path",
        "native_t2_path",
        "multiclass_mask_path",
    ]
    for column in path_columns:
        usable = usable[usable[column].astype(str).ne("")]
    usable["union_voxels"] = usable[["label1_voxels", "label2_voxels", "label3_voxels"]].fillna(0).sum(axis=1)
    usable = usable[usable["union_voxels"] > 0].copy()

    usable["progression_day"] = usable[
        "clinical_number_of_days_from_diagnosis_to_date_of_first_progression"
    ].fillna(usable["clinical_time_to_first_progression_days"])
    usable["label"] = 0.0
    progressed = usable["clinical_progression"].eq(1) & usable["progression_day"].notna() & usable["days_from_diagnosis_to_mri"].notna()
    usable.loc[progressed, "label"] = (
        (usable.loc[progressed, "progression_day"] - usable.loc[progressed, "days_from_diagnosis_to_mri"]) <= 120
    ).astype(float)
    usable = usable[usable["label"].isin([0.0, 1.0])].copy()
    usable["label"] = usable["label"].astype(int)

    late_treatment_cols = (
        "clinical_number_of_days_from_diagnosis_to_starting_2nd_additional_therapy",
        "clinical_number_of_days_from_diagnosis_to_start_immunotherapy",
        "clinical_days_from_diagnosis_to_new_treatment",
    )
    usable["late_treatment_start_day"] = earliest_nonnegative_day(usable, late_treatment_cols)
    usable = usable[
        usable["late_treatment_start_day"].isna()
        | (
            usable["days_from_diagnosis_to_mri"].notna()
            & (usable["days_from_diagnosis_to_mri"] < usable["late_treatment_start_day"])
        )
    ].copy()

    pre_progression_mask = usable["clinical_progression"].eq(0) | (
        usable["progression_day"].notna()
        & usable["days_from_diagnosis_to_mri"].notna()
        & (usable["days_from_diagnosis_to_mri"] < usable["progression_day"])
    )
    usable = usable[pre_progression_mask].copy()
    return usable


def classify_column(name: str) -> tuple[str, str]:
    if name in MOLECULAR_FEATURE_COLUMNS or name in HYBRID_BASIC_EXTRA_COLUMNS or name in ENGINEERED_CATEGORICAL_COLUMNS:
        return "used_currently", "already used by current hybrid feature sets"
    if name == "clinical_progression":
        return "target", "defines whether the patient ever progressed"
    if re.search(r"time_to_first_progression|date_of_first_progression|further_progression", name):
        return "leakage_high", "directly encodes outcome timing"
    if re.search("|".join(LEAKY_NAME_PATTERNS), name):
        return "leakage_high", "appears to encode post-outcome status or later treatment trajectory"
    if re.search(r"number_of_days_from_diagnosis_to_\\d(st|nd|rd|th)_mri_timepoint", name):
        return "duplicate_timing", "encodes scan schedule rather than baseline biology"
    if re.search(r"number_of_days_from_diagnosis_to", name):
        if any(token in name for token in ("first_surgery_or_procedure", "radiation_therapy_start", "initial_chemo_therapy_start_date")):
            return "candidate_timing", "pre-scan treatment timing that may be usable with careful leakage review"
        return "timing_unclear", "timeline variable needs manual review before modeling"
    if re.search(r"initial_chemo_therapy|radiation_therapy|dose|fractions|stereotactic_biopsy|multiple_surgeries|previous_brain_tumor|type_of_previous_brain_tumor|year_of_previous_surgery", name):
        return "candidate_clinical", "potentially usable pre-scan clinical context"
    return "candidate_clinical", "appears clinically usable pending coding review"


def summarize_column(series: pd.Series) -> dict[str, object]:
    non_null = int(series.notna().sum())
    total = int(len(series))
    completeness = float(non_null / total) if total else float("nan")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_non_null = int(numeric.notna().sum())
    is_numeric = numeric_non_null == non_null and non_null > 0
    if is_numeric:
        unique_values = sorted(numeric.dropna().unique().tolist())
        preview = unique_values[:10]
        return {
            "dtype_family": "numeric",
            "non_null": non_null,
            "completeness": completeness,
            "n_unique_non_null": int(numeric.dropna().nunique()),
            "value_preview": json.dumps(preview),
            "min_value": float(numeric.min()) if numeric_non_null else np.nan,
            "median_value": float(numeric.median()) if numeric_non_null else np.nan,
            "max_value": float(numeric.max()) if numeric_non_null else np.nan,
        }
    cleaned = series.fillna("missing").astype(str).str.strip().replace({"": "missing", "nan": "missing", "None": "missing"})
    value_counts = cleaned.value_counts(dropna=False).head(10).to_dict()
    return {
        "dtype_family": "categorical",
        "non_null": non_null,
        "completeness": completeness,
        "n_unique_non_null": int(cleaned[cleaned != "missing"].nunique()),
        "value_preview": json.dumps(value_counts, sort_keys=True),
        "min_value": np.nan,
        "median_value": np.nan,
        "max_value": np.nan,
    }


def patient_constancy(df: pd.DataFrame, column: str) -> float:
    patient_uniques = (
        df[["patient_id", column]]
        .dropna(subset=[column])
        .groupby("patient_id")[column]
        .nunique()
    )
    if patient_uniques.empty:
        return float("nan")
    return float((patient_uniques <= 1).mean())


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(args.experiment_index.resolve())
    df = build_forward_cohort(index_df) if args.forward_cohort_only else index_df.copy()
    clinical_columns = [column for column in df.columns if column.startswith("clinical_")]

    rows: list[dict[str, object]] = []
    for column in clinical_columns:
        label = classify_column(column)
        summary = summarize_column(df[column])
        rows.append(
            {
                "column": column,
                "audit_group": label[0],
                "audit_note": label[1],
                "patient_constancy": patient_constancy(df, column),
                **summary,
            }
        )

    audit = pd.DataFrame(rows).sort_values(
        ["audit_group", "completeness", "n_unique_non_null", "column"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    audit.to_csv(output_dir / "clinical_feature_audit.csv", index=False)

    grouped = audit.groupby("audit_group")["column"].count().sort_values(ascending=False)
    payload = {
        "rows": int(len(df)),
        "patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else None,
        "forward_cohort_only": bool(args.forward_cohort_only),
        "group_counts": {str(k): int(v) for k, v in grouped.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Clinical Feature Audit",
        "",
        f"- Rows audited: `{payload['rows']}`",
        f"- Patients audited: `{payload['patients']}`",
        f"- Forward cohort only: `{payload['forward_cohort_only']}`",
        "",
        "## Group Counts",
        "",
    ]
    for group, count in payload["group_counts"].items():
        lines.append(f"- `{group}`: `{count}`")

    lines.extend(
        [
            "",
            "## Top Candidate Clinical Fields",
            "",
        ]
    )
    candidates = audit[audit["audit_group"].isin(["candidate_clinical", "candidate_timing"])].head(20)
    for row in candidates.itertuples(index=False):
        lines.append(
            f"- `{row.column}`: completeness `{row.completeness:.3f}`, unique `{row.n_unique_non_null}`, patient constancy `{row.patient_constancy:.3f}`; {row.audit_note}"
        )

    lines.extend(
        [
            "",
            "## Highest Leakage-Risk Fields",
            "",
        ]
    )
    leakage = audit[audit["audit_group"].isin(["target", "leakage_high", "duplicate_timing", "timing_unclear"])].head(25)
    for row in leakage.itertuples(index=False):
        lines.append(
            f"- `{row.column}`: completeness `{row.completeness:.3f}`; {row.audit_note}"
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
