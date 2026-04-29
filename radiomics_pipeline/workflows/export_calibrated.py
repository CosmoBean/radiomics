#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from radiomics_pipeline.workflows import train as surv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the calibrated surveillance model bundle from an existing result directory."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=REPO_ROOT / "results" / "repeated_forward_hybrid_basic_corrected" / "seed_62",
        help="Result directory containing summary.json, radiomics_features.csv, and test/train patient files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "models" / "calibrated",
        help="Directory where the exported bundle and metadata should be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=62,
        help="Seed used for the original run.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="CV folds used when refitting the calibration stack.",
    )
    return parser.parse_args(argv)


def load_inputs(result_dir: Path) -> tuple[dict[str, object], pd.DataFrame, set[str], set[str]]:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    features = pd.read_csv(result_dir / "radiomics_features.csv")
    test_patients = {
        line.strip()
        for line in (result_dir / "test_patients.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    train_patients = {
        line.strip()
        for line in (result_dir / "train_patients.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return summary, features, test_patients, train_patients


def modality_for_feature(feature: str) -> str:
    if feature.startswith(surv.CLINICAL_FEATURE_PREFIX):
        return "clinical"
    if feature.startswith(surv.ENGINEERED_FEATURE_PREFIX):
        return "engineered"
    return feature.split("_", 1)[0]


def make_args(output_dir: Path, seed: int, cv_folds: int) -> argparse.Namespace:
    return argparse.Namespace(
        cv_folds=cv_folds,
        seed=seed,
        lightgbm_device="cpu",
        output_dir=output_dir,
    )


def export_bundle(
    summary: dict[str, object],
    features: pd.DataFrame,
    test_patients: set[str],
    result_dir: Path,
    output_dir: Path,
    seed: int,
    cv_folds: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model = dict(summary["best_model"])
    model_name = str(best_model["model"])
    columns = list(best_model["columns"])
    params = dict(best_model["params"])
    params["variance_threshold"] = float(params["variance_threshold"])
    params["corr_threshold"] = float(params["corr_threshold"])

    features = features.copy()
    features["split"] = np.where(features["patient_id"].isin(test_patients), "test", "train")
    train_df = features[features["split"] == "train"].copy()
    test_df = features[features["split"] == "test"].copy()

    args = make_args(output_dir=output_dir, seed=seed, cv_folds=cv_folds)
    oof_raw, fold_rows, final_state, final_model = surv.fit_oof_predictions(
        train_df=train_df,
        model_name=model_name,
        params=params,
        columns=columns,
        args=args,
    )

    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
    calibrator.fit(oof_raw.reshape(-1, 1), train_df["label"])
    oof_cal = calibrator.predict_proba(oof_raw.reshape(-1, 1))[:, 1]
    threshold, threshold_stats = surv.select_threshold(train_df["label"].to_numpy(), oof_cal)

    transformed_test = surv.transform_preprocessor(test_df[columns], final_state)
    test_raw = surv.predict_positive_probability(final_model, transformed_test)
    test_cal = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]

    bundle = {
        "model_name": model_name,
        "selected_columns": columns,
        "preprocessor_state": final_state,
        "classifier": final_model,
        "calibrator": calibrator,
        "threshold": threshold,
        "seed": seed,
        "cv_folds": cv_folds,
        "modalities": summary.get("modalities", []),
        "clinical_feature_set": summary.get("clinical_feature_set"),
    }
    with (output_dir / "model_bundle.pkl").open("wb") as fh:
        pickle.dump(bundle, fh)

    metadata = {
        "source_result_dir": str(result_dir.resolve()),
        "seed": seed,
        "cv_folds": cv_folds,
        "model_name": model_name,
        "clinical_feature_set": summary.get("clinical_feature_set"),
        "modalities": summary.get("modalities", []),
        "selected_feature_count": len(columns),
        "selected_columns": columns,
        "threshold": float(threshold),
        "threshold_stats": threshold_stats,
        "test_metrics_raw": surv.threshold_metrics(test_df["label"].to_numpy(), test_raw, threshold),
        "test_metrics_calibrated": surv.threshold_metrics(test_df["label"].to_numpy(), test_cal, threshold),
        "test_brier_raw": float(surv.brier_score_loss(test_df["label"], test_raw)),
        "test_brier_calibrated": float(surv.brier_score_loss(test_df["label"], test_cal)),
        "cv_fold_metrics": fold_rows,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

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
    predictions["predicted_class_by_threshold"] = (test_cal >= threshold).astype(int)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    transformed_train = surv.transform_preprocessor(train_df[columns], final_state)
    explainer = shap.LinearExplainer(final_model, transformed_train)
    shap_values = explainer.shap_values(transformed_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_array = np.asarray(shap_values, dtype=float)
    shap_df = pd.DataFrame(
        {
            "feature": transformed_test.columns,
            "mean_abs_shap": np.abs(shap_array).mean(axis=0),
            "modality": [modality_for_feature(feature) for feature in transformed_test.columns],
        }
    ).sort_values("mean_abs_shap", ascending=False, kind="stable")
    shap_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    modality_share = (
        shap_df.groupby("modality", as_index=False)["mean_abs_shap"].sum().sort_values("mean_abs_shap", ascending=False)
    )
    if modality_share["mean_abs_shap"].sum() > 0:
        modality_share["share"] = modality_share["mean_abs_shap"] / modality_share["mean_abs_shap"].sum()
    else:
        modality_share["share"] = 0.0
    modality_share.to_csv(output_dir / "shap_modality_share.csv", index=False)

    return metadata, shap_df


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary, features, test_patients, _ = load_inputs(args.result_dir)
    export_bundle(
        summary=summary,
        features=features,
        test_patients=test_patients,
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
