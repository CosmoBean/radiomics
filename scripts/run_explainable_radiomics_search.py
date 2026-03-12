#!/usr/bin/env python3
"""Run a bounded auto-search over explainable radiomics models."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODALITIES = ("t1", "t1c", "flair", "t2")
FAMILY_MAP = {
    "shape": ("shape_",),
    "firstorder": ("firstorder_",),
    "glcm": ("glcm_",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search explainable radiomics configurations with an iteration and time budget."
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
        default=Path("results/explainable_radiomics_grade4_auto"),
        help="Directory for search outputs.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="grade4_vs_lower",
        choices=["grade4_vs_lower", "gbm_vs_other", "overall_survival_death"],
        help="Derived binary target to model.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=25,
        help="Maximum number of searched configurations.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=3600,
        help="Maximum wall-clock runtime.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260310,
        help="Random seed for reproducible search.",
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


def candidate_feature_columns(
    all_feature_cols: list[str],
    modality_subset: tuple[str, ...],
    family_subset: tuple[str, ...],
) -> list[str]:
    prefixes = tuple(f"{modality}_" for modality in modality_subset)
    family_tokens = tuple(token for family in family_subset for token in FAMILY_MAP[family])
    cols = [
        col
        for col in all_feature_cols
        if col.startswith(prefixes) and any(token in col for token in family_tokens)
    ]
    return cols


def sample_config(rng: np.random.Generator) -> dict[str, object]:
    modality_count = int(rng.integers(1, len(MODALITIES) + 1))
    modality_subset = tuple(sorted(rng.choice(MODALITIES, size=modality_count, replace=False).tolist()))
    families = list(FAMILY_MAP.keys())
    family_count = int(rng.integers(1, len(families) + 1))
    family_subset = tuple(sorted(rng.choice(families, size=family_count, replace=False).tolist()))
    k = int(rng.choice([4, 6, 8, 10, 12, 16, 20, 24]))
    c_value = float(10 ** rng.uniform(-2.5, 0.7))
    penalty = str(rng.choice(["l1", "l2"]))
    class_weight = str(rng.choice(["balanced", "none"]))
    return {
        "modalities": modality_subset,
        "families": family_subset,
        "k": k,
        "C": c_value,
        "penalty": penalty,
        "class_weight": None if class_weight == "none" else class_weight,
    }


def build_pipeline(config: dict[str, object], k_eff: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k_eff)),
            (
                "model",
                LogisticRegression(
                    penalty=str(config["penalty"]),
                    solver="liblinear",
                    C=float(config["C"]),
                    max_iter=5000,
                    class_weight=config["class_weight"],
                    random_state=20260310,
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
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
    all_feature_cols = [column for column in merged.columns if column not in metadata_cols]

    train_df = merged[merged["split"] == "train"].copy()
    val_df = merged[merged["split"] == "val"].copy()
    test_df = merged[merged["split"] == "test"].copy()

    y_train = train_df["target"]
    y_val = val_df["target"]
    y_test = test_df["target"]

    start_time = time.time()
    trials: list[dict[str, object]] = []
    best_key = None
    best_pipeline = None
    best_columns: list[str] = []
    best_trial: dict[str, object] | None = None

    for iteration in range(1, args.max_iters + 1):
        elapsed = time.time() - start_time
        if elapsed >= args.max_runtime_seconds:
            break

        config = sample_config(rng)
        candidate_cols = candidate_feature_columns(
            all_feature_cols=all_feature_cols,
            modality_subset=config["modalities"],
            family_subset=config["families"],
        )
        if not candidate_cols:
            continue

        X_train = train_df[candidate_cols]
        X_val = val_df[candidate_cols]
        X_test = test_df[candidate_cols]
        k_eff = min(int(config["k"]), len(candidate_cols))
        pipeline = build_pipeline(config=config, k_eff=k_eff)
        pipeline.fit(X_train, y_train)

        split_metrics = {}
        for split_name, X_split, y_split in (
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ):
            prob = pipeline.predict_proba(X_split)[:, 1]
            pred = pipeline.predict(X_split)
            split_metrics[split_name] = metric_block(y_split, pred, prob)

        nonzero_coefficients = int(np.count_nonzero(pipeline.named_steps["model"].coef_.ravel()))
        trial = {
            "iteration": iteration,
            "elapsed_seconds": round(time.time() - start_time, 3),
            "modalities": "|".join(config["modalities"]),
            "families": "|".join(config["families"]),
            "k": int(config["k"]),
            "effective_k": int(k_eff),
            "C": float(config["C"]),
            "penalty": str(config["penalty"]),
            "class_weight": "balanced" if config["class_weight"] else "none",
            "candidate_feature_count": int(len(candidate_cols)),
            "nonzero_coefficients": nonzero_coefficients,
            "train_accuracy": split_metrics["train"]["accuracy"],
            "train_balanced_accuracy": split_metrics["train"]["balanced_accuracy"],
            "train_roc_auc": split_metrics["train"]["roc_auc"],
            "val_accuracy": split_metrics["val"]["accuracy"],
            "val_balanced_accuracy": split_metrics["val"]["balanced_accuracy"],
            "val_roc_auc": split_metrics["val"]["roc_auc"],
            "test_accuracy": split_metrics["test"]["accuracy"],
            "test_balanced_accuracy": split_metrics["test"]["balanced_accuracy"],
            "test_roc_auc": split_metrics["test"]["roc_auc"],
        }
        trials.append(trial)

        key = (
            trial["val_balanced_accuracy"],
            trial["val_roc_auc"],
            -trial["nonzero_coefficients"],
            -trial["effective_k"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_pipeline = pipeline
            best_columns = candidate_cols
            best_trial = trial

        print(
            f"iter {iteration}/{args.max_iters} elapsed={trial['elapsed_seconds']}s "
            f"val_bal_acc={trial['val_balanced_accuracy']:.4f} test_bal_acc={trial['test_balanced_accuracy']:.4f} "
            f"modalities={trial['modalities']} families={trial['families']} k={trial['effective_k']} "
            f"penalty={trial['penalty']} C={trial['C']:.4f}",
            flush=True,
        )

    if not trials or best_pipeline is None or best_trial is None:
        raise SystemExit("No valid search trials completed.")

    trials_df = pd.DataFrame(trials).sort_values(
        ["val_balanced_accuracy", "val_roc_auc", "nonzero_coefficients", "effective_k"],
        ascending=[False, False, True, True],
        kind="stable",
    )

    support = best_pipeline.named_steps["select"].get_support(indices=True)
    selected_features = [best_columns[idx] for idx in support]
    coefficients = best_pipeline.named_steps["model"].coef_.ravel()
    explainability = pd.DataFrame(
        {
            "feature": selected_features,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "direction": [
                "higher -> positive_class" if value > 0 else "higher -> negative_class"
                for value in coefficients
            ],
        }
    ).sort_values("coefficient", ascending=False, kind="stable")

    X_test_best = test_df[best_columns]
    test_prob = best_pipeline.predict_proba(X_test_best)[:, 1]
    test_pred = best_pipeline.predict(X_test_best)
    test_predictions = test_df[["patient_id", "timepoint", "split", "target"]].copy()
    test_predictions["predicted_class"] = test_pred
    test_predictions["predicted_probability"] = test_prob

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(output_dir / "search_iterations.csv", index=False)
    explainability.to_csv(output_dir / "best_coefficients.csv", index=False)
    test_predictions.to_csv(output_dir / "best_test_predictions.csv", index=False)

    summary = {
        "target": args.target,
        "max_iters": args.max_iters,
        "max_runtime_seconds": args.max_runtime_seconds,
        "completed_iterations": int(len(trials_df)),
        "elapsed_seconds": float(round(time.time() - start_time, 3)),
        "num_patients": int(len(merged)),
        "train_patients": int(len(train_df)),
        "val_patients": int(len(val_df)),
        "test_patients": int(len(test_df)),
        "best_trial": best_trial,
        "selected_features": selected_features,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {output_dir / 'search_iterations.csv'}")
    print(f"Wrote {output_dir / 'best_coefficients.csv'}")
    print(f"Wrote {output_dir / 'best_test_predictions.csv'}")


if __name__ == "__main__":
    main()
