#!/usr/bin/env python3
"""Compute subgroup representation and performance metrics for model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


DEFAULT_GROUP_COLUMNS = (
    "clinical_sex_at_birth",
    "clinical_race",
    "clinical_grade_of_primary_brain_tumor",
    "clinical_primary_diagnosis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze subgroup representation and predictive performance."
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("results/test_predictions.csv"),
        help="Predictions CSV with patient_id/timepoint keys and probabilities.",
    )
    parser.add_argument(
        "--experiment-index",
        type=Path,
        default=Path("processed/manifests/experiment_index.csv"),
        help="Experiment index with clinical covariates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bias_analysis"),
        help="Directory for subgroup reports.",
    )
    parser.add_argument(
        "--probability-column",
        type=str,
        default="calibrated_probability",
        help="Probability column to score.",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="predicted_class",
        help="Predicted class column.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Ground-truth label column.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=5,
        help="Minimum subgroup size to include in the report.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_category(series: pd.Series) -> pd.Series:
    values = series.fillna("missing").astype(str).str.strip()
    return values.replace({"": "missing", "nan": "missing", "None": "missing"})


def build_age_group(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    bins = [-np.inf, 39, 49, 59, 69, np.inf]
    labels = ["<40", "40-49", "50-59", "60-69", "70+"]
    grouped = pd.cut(numeric, bins=bins, labels=labels)
    return normalize_category(grouped.astype(object))


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(brier_score_loss(y_true, y_prob))


def subgroup_metrics(df: pd.DataFrame, label_col: str, pred_col: str, prob_col: str) -> dict[str, float | int]:
    y_true = df[label_col].to_numpy(dtype=int)
    y_pred = df[pred_col].to_numpy(dtype=int)
    y_prob = df[prob_col].to_numpy(dtype=float)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")

    return {
        "n": int(len(df)),
        "positives": int(y_true.sum()),
        "negative": int((y_true == 0).sum()),
        "prevalence": float(y_true.mean()) if len(df) else float("nan"),
        "predicted_positive_rate": float(y_pred.mean()) if len(df) else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(df) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if np.unique(y_true).size == 2 else float("nan"),
        "roc_auc": safe_auc(y_true, y_prob),
        "brier": safe_brier(y_true, y_prob),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def markdown_report(summary: dict[str, object], details: pd.DataFrame) -> str:
    lines = [
        "# Bias Analysis",
        "",
        "## Overview",
        f"- Samples analyzed: `{summary['samples']}`",
        f"- Patients analyzed: `{summary['patients']}`",
        f"- Label column: `{summary['label_column']}`",
        f"- Probability column: `{summary['probability_column']}`",
        f"- Prediction column: `{summary['prediction_column']}`",
        "",
        "## Largest Subgroup Gaps",
    ]

    for group_name in summary["largest_balanced_accuracy_gaps"]:
        subset = details[details["group"] == group_name].copy()
        if subset.empty:
            continue
        gap = subset["balanced_accuracy"].max() - subset["balanced_accuracy"].min()
        lines.append(f"- `{group_name}` balanced-accuracy gap: `{gap:.4f}`")

    lines.extend(["", "## Included Groups"])
    for group_name in sorted(details["group"].unique().tolist()):
        subset = details[details["group"] == group_name].copy()
        labels = ", ".join(
            f"{row.subgroup} (n={int(row.n)}, auc={row.roc_auc if pd.notna(row.roc_auc) else float('nan'):.3f})"
            for row in subset.itertuples(index=False)
        )
        lines.append(f"- `{group_name}`: {labels}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions_csv)
    experiment_index = pd.read_csv(args.experiment_index)

    required_prediction_cols = {
        "patient_id",
        "timepoint",
        args.label_column,
        args.prediction_column,
        args.probability_column,
    }
    missing_prediction_cols = sorted(required_prediction_cols - set(predictions.columns))
    if missing_prediction_cols:
        raise SystemExit(f"Predictions CSV is missing required columns: {missing_prediction_cols}")

    merge_columns = [
        "patient_id",
        "timepoint",
        "timepoint_number",
        "clinical_age_at_diagnosis",
        *[column for column in DEFAULT_GROUP_COLUMNS if column in experiment_index.columns],
    ]
    merged = predictions.merge(
        experiment_index[sorted(set(merge_columns))],
        on=["patient_id", "timepoint"],
        how="left",
        validate="one_to_one",
    )
    merged["age_group"] = build_age_group(merged.get("clinical_age_at_diagnosis", pd.Series(index=merged.index, dtype=float)))

    group_frames: list[pd.DataFrame] = []
    candidate_groups = [column for column in DEFAULT_GROUP_COLUMNS if column in merged.columns] + ["age_group"]
    for group_name in candidate_groups:
        subgroup_series = normalize_category(merged[group_name])
        for subgroup, subgroup_df in merged.groupby(subgroup_series, sort=True):
            if len(subgroup_df) < args.min_group_size:
                continue
            row = subgroup_metrics(
                subgroup_df,
                label_col=args.label_column,
                pred_col=args.prediction_column,
                prob_col=args.probability_column,
            )
            row["group"] = group_name
            row["subgroup"] = subgroup
            group_frames.append(pd.DataFrame([row]))

    if not group_frames:
        raise SystemExit("No subgroup met the minimum size threshold.")

    details = pd.concat(group_frames, ignore_index=True)
    details = details.sort_values(["group", "n", "subgroup"], ascending=[True, False, True], kind="stable")

    balanced_accuracy_gaps: dict[str, float] = {}
    for group_name, subset in details.groupby("group"):
        valid = subset["balanced_accuracy"].dropna()
        if not valid.empty:
            balanced_accuracy_gaps[group_name] = float(valid.max() - valid.min())

    summary = {
        "samples": int(len(merged)),
        "patients": int(merged["patient_id"].nunique()),
        "label_column": args.label_column,
        "prediction_column": args.prediction_column,
        "probability_column": args.probability_column,
        "min_group_size": int(args.min_group_size),
        "groups_analyzed": sorted(details["group"].unique().tolist()),
        "largest_balanced_accuracy_gaps": [
            name
            for name, _ in sorted(
                balanced_accuracy_gaps.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "balanced_accuracy_gaps": balanced_accuracy_gaps,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(markdown_report(summary, details), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
