#!/usr/bin/env python3
"""Run an explainable radiomics model on cached extracted features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit an explainable radiomics model on a derived binary target."
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("results/radiomics_baseline_progression/radiomics_features.csv"),
        help="Cached radiomics feature table.",
    )
    parser.add_argument(
        "--experiment-index",
        type=Path,
        default=Path("processed/manifests/experiment_index.csv"),
        help="Merged experiment index for deriving labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/explainable_radiomics_grade4"),
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="grade4_vs_lower",
        choices=["grade4_vs_lower", "gbm_vs_other", "overall_survival_death"],
        help="Derived binary target to model.",
    )
    return parser.parse_args()


def build_target(df: pd.DataFrame, target_name: str) -> pd.Series:
    if target_name == "grade4_vs_lower":
        grade = df["clinical_grade_of_primary_brain_tumor"].astype(str)
        target = pd.Series(np.nan, index=df.index, dtype=float)
        target.loc[grade == "4"] = 1.0
        target.loc[grade.isin(["1", "2", "3"])] = 0.0
        return target
    if target_name == "gbm_vs_other":
        diagnosis = df["clinical_primary_diagnosis"].astype(str)
        return diagnosis.eq("GBM").astype(float)
    if target_name == "overall_survival_death":
        return df["clinical_overall_survival_death"].astype(float)
    raise ValueError(f"Unsupported target: {target_name}")


def metric_block(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features_csv)
    index_df = pd.read_csv(args.experiment_index)
    earliest = index_df.sort_values(["patient_id", "timepoint_number"], kind="stable").groupby("patient_id", as_index=False).first()
    merged = features.merge(
        earliest[
            [
                "patient_id",
                "timepoint",
                "split",
                "clinical_grade_of_primary_brain_tumor",
                "clinical_primary_diagnosis",
                "clinical_overall_survival_death",
            ]
        ],
        on=["patient_id", "timepoint", "split"],
        how="left",
    )
    merged["target"] = build_target(merged, args.target)
    merged = merged[merged["target"].notna()].copy()
    merged["target"] = merged["target"].astype(int)

    metadata_cols = [
        "patient_id",
        "timepoint",
        "split",
        "target",
        "clinical_grade_of_primary_brain_tumor",
        "clinical_primary_diagnosis",
        "clinical_overall_survival_death",
    ]
    feature_cols = [column for column in merged.columns if column not in metadata_cols]

    train_df = merged[merged["split"] == "train"].copy()
    val_df = merged[merged["split"] == "val"].copy()
    test_df = merged[merged["split"] == "test"].copy()

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    best = None
    best_pipeline = None
    best_result = None
    for k in (8, 12, 16):
        k_eff = min(k, X_train.shape[1])
        for c in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("var", VarianceThreshold()),
                    ("scale", StandardScaler()),
                    ("select", SelectKBest(score_func=f_classif, k=k_eff)),
                    (
                        "model",
                        LogisticRegression(
                            penalty="l1",
                            solver="liblinear",
                            C=c,
                            max_iter=5000,
                            class_weight="balanced",
                            random_state=20260310,
                        ),
                    ),
                ]
            )
            pipeline.fit(X_train, y_train)
            val_prob = pipeline.predict_proba(X_val)[:, 1]
            val_pred = pipeline.predict(X_val)
            val_metrics = metric_block(y_val, val_pred, val_prob)
            candidate = (val_metrics["balanced_accuracy"], val_metrics["roc_auc"], -k_eff, -c)
            if best is None or candidate > best:
                best = candidate
                best_pipeline = pipeline
                best_result = {"k": k_eff, "C": c, "val_metrics": val_metrics}

    assert best_pipeline is not None and best_result is not None

    results = {}
    for split_name, X_split, y_split in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        prob = best_pipeline.predict_proba(X_split)[:, 1]
        pred = best_pipeline.predict(X_split)
        results[split_name] = metric_block(y_split, pred, prob)

    majority_class = int(y_train.mode().iloc[0])
    majority_results = {}
    for split_name, y_split in (("train", y_train), ("val", y_val), ("test", y_test)):
        pred = np.full(len(y_split), majority_class)
        prob = np.full(len(y_split), majority_class, dtype=float)
        majority_results[split_name] = metric_block(y_split, pred, prob)

    selector = best_pipeline.named_steps["select"]
    model = best_pipeline.named_steps["model"]
    support = selector.get_support(indices=True)
    selected_features = [feature_cols[idx] for idx in support]
    coefficients = model.coef_.ravel()
    coefficient_df = pd.DataFrame(
        {
            "feature": selected_features,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "direction": ["higher -> positive_class" if value > 0 else "higher -> negative_class" for value in coefficients],
        }
    ).sort_values("coefficient", ascending=False, kind="stable")

    test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    test_pred = best_pipeline.predict(X_test)
    test_predictions = test_df[["patient_id", "timepoint", "split", "target"]].copy()
    test_predictions["predicted_class"] = test_pred
    test_predictions["predicted_probability"] = test_prob

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficient_df.to_csv(output_dir / "explainability_coefficients.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "target": args.target,
        "num_patients": int(len(merged)),
        "train_patients": int(len(train_df)),
        "val_patients": int(len(val_df)),
        "test_patients": int(len(test_df)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_val": float(y_val.mean()),
        "positive_rate_test": float(y_test.mean()),
        "selected_k": int(best_result["k"]),
        "selected_C": float(best_result["C"]),
        "train_metrics": results["train"],
        "val_metrics": results["val"],
        "test_metrics": results["test"],
        "majority_baseline": majority_results,
        "selected_features": selected_features,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {output_dir / 'explainability_coefficients.csv'}")
    print(f"Wrote {output_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
