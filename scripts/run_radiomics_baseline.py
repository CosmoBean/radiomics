#!/usr/bin/env python3
"""Train statistical ML baselines on radiomics features."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor, logger as radiomics_logger
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


MODALITIES = ("t1", "t1c", "flair", "t2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run radiomics baselines on earliest available MRI per patient."
    )
    parser.add_argument(
        "--experiment-index",
        type=Path,
        default=Path("processed/manifests/experiment_index.csv"),
        help="Merged experiment index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/radiomics_baseline"),
        help="Directory for extracted features and metrics.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="clinical_progression",
        help="Binary label column from experiment_index.csv.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=32,
        help="Maximum number of selected features.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for radiomics extraction.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract radiomics features and skip model fitting.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Recompute radiomics features even if a cached CSV is present.",
    )
    return parser.parse_args()


def build_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=25,
        normalize=False,
        resampledPixelSpacing=None,
        interpolator=sitk.sitkBSpline,
        correctMask=True,
    )
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("shape")
    return extractor


def repo_root() -> Path:
    return Path.cwd().resolve()


def path_or_empty(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def extract_case_features(
    row: dict[str, object],
    root: Path,
) -> dict[str, object]:
    extractor = build_extractor()
    record: dict[str, object] = {
        "patient_id": row["patient_id"],
        "timepoint": row["timepoint"],
        "split": row["split"],
        "target": int(row["target"]),
    }

    mask_path = root / row["roi_binary_mask_path"]
    mask_image = sitk.ReadImage(str(mask_path))

    for modality in MODALITIES:
        image_path = path_or_empty(row[f"roi_{modality}_path"])
        if not image_path:
            continue
        image = sitk.ReadImage(str(root / image_path))
        features = extractor.execute(image, mask_image, label=1)
        for name, value in features.items():
            if name.startswith("diagnostics_"):
                continue
            key = f"{modality}_{name.replace('original_', '')}"
            record[key] = float(value)

    return record


def extract_case_features_worker(row: dict[str, object], root_str: str) -> dict[str, object]:
    warnings.filterwarnings("ignore")
    radiomics_logger.setLevel(logging.ERROR)
    return extract_case_features(row=row, root=Path(root_str))


def select_earliest_cases(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    usable = df[(df["roi_status"] == "written") & df[target_column].notna()].copy()
    usable = usable.sort_values(["patient_id", "timepoint_number"], kind="stable")
    earliest = usable.groupby("patient_id", as_index=False).first()
    earliest["target"] = earliest[target_column].astype(int)
    return earliest


def evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_features: int,
) -> tuple[dict[str, object], Pipeline]:
    k = min(max_features, X_train.shape[1]) if X_train.shape[1] else "all"
    pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k if k != 0 else "all")),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)

    result: dict[str, object] = {"model": name}
    for split_name, X_split, y_split in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        pred = pipeline.predict(X_split)
        result[f"{split_name}_accuracy"] = float(accuracy_score(y_split, pred))
        result[f"{split_name}_balanced_accuracy"] = float(balanced_accuracy_score(y_split, pred))
        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(X_split)[:, 1]
        else:
            score = pipeline.decision_function(X_split)
            prob = 1.0 / (1.0 + np.exp(-score))
        result[f"{split_name}_roc_auc"] = float(roc_auc_score(y_split, prob))
    result["selected_feature_count"] = int(
        pipeline.named_steps["select"].get_support(indices=True).shape[0]
    )
    return result, pipeline


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")
    radiomics_logger.setLevel(logging.ERROR)
    root = repo_root()
    df = pd.read_csv(args.experiment_index)
    cases = select_earliest_cases(df, args.target_column)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features_csv = output_dir / "radiomics_features.csv"

    if features_csv.exists() and not args.force_extract:
        features = pd.read_csv(features_csv)
        print(f"Using cached radiomics features from {features_csv}")
    else:
        case_records = cases.to_dict(orient="records")
        feature_rows: list[dict[str, object]] = []
        total = len(case_records)

        if args.workers <= 1:
            for index, case_row in enumerate(case_records, start=1):
                feature_rows.append(extract_case_features(case_row, root))
                if index % 25 == 0 or index == total:
                    print(f"Extracted radiomics for {index}/{total} cases", flush=True)
        else:
            max_workers = min(args.workers, os.cpu_count() or args.workers)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(extract_case_features_worker, case_row, str(root))
                    for case_row in case_records
                ]
                for index, future in enumerate(as_completed(futures), start=1):
                    feature_rows.append(future.result())
                    if index % 25 == 0 or index == total:
                        print(f"Extracted radiomics for {index}/{total} cases", flush=True)

        features = pd.DataFrame(feature_rows).sort_values(["patient_id"], kind="stable")
        features.to_csv(features_csv, index=False)
        print(f"Wrote {features_csv}")

    metadata_cols = ["patient_id", "timepoint", "split", "target"]
    feature_cols = [column for column in features.columns if column not in metadata_cols]

    if args.extract_only:
        summary = {
            "target_column": args.target_column,
            "task_definition": "earliest available ROI-enabled MRI per patient",
            "num_patients": int(len(features)),
            "num_features": int(len(feature_cols)),
            "workers": args.workers,
            "extract_only": True,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    train_df = features[features["split"] == "train"].copy()
    val_df = features[features["split"] == "val"].copy()
    test_df = features[features["split"] == "test"].copy()

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    models = [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
                random_state=20260310,
            ),
        ),
        (
            "linear_svm",
            SVC(
                kernel="linear",
                class_weight="balanced",
                probability=True,
                random_state=20260310,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=20260310,
            ),
        ),
    ]

    results: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}
    for name, model in models:
        metrics, fitted = evaluate_model(
            name=name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            max_features=args.max_features,
        )
        results.append(metrics)
        fitted_models[name] = fitted

    results_df = pd.DataFrame(results).sort_values("val_balanced_accuracy", ascending=False, kind="stable")
    best_model_name = str(results_df.iloc[0]["model"])
    best_pipeline = fitted_models[best_model_name]

    selected_idx = best_pipeline.named_steps["select"].get_support(indices=True)
    selected_features = [feature_cols[idx] for idx in selected_idx]
    selection_scores = best_pipeline.named_steps["select"].scores_
    feature_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "score": selection_scores,
            "selected": [feature in selected_features for feature in feature_cols],
        }
    ).sort_values(["selected", "score"], ascending=[False, False], kind="stable")

    results_df.to_csv(output_dir / "model_metrics.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_scores.csv", index=False)

    summary = {
        "target_column": args.target_column,
        "task_definition": "earliest available ROI-enabled MRI per patient",
        "num_patients": int(len(features)),
        "num_features": int(len(feature_cols)),
        "train_patients": int(len(train_df)),
        "val_patients": int(len(val_df)),
        "test_patients": int(len(test_df)),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
        "best_model": best_model_name,
        "best_model_metrics": results_df.iloc[0].to_dict(),
        "selected_features": selected_features,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {output_dir / 'radiomics_features.csv'}")
    print(f"Wrote {output_dir / 'model_metrics.csv'}")
    print(f"Wrote {output_dir / 'feature_scores.csv'}")


if __name__ == "__main__":
    main()
