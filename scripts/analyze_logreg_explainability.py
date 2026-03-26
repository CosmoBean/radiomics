#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_postoperative_progression_surveillance as surv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a coefficient and stability report for a fitted logistic surveillance model."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Result directory containing summary.json, radiomics_features.csv, and test_patients.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for explainability outputs. Defaults to <result-dir>/explainability.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Patient-aware CV folds for coefficient stability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV splitting.",
    )
    return parser.parse_args()


def feature_group(feature: str) -> str:
    if feature.startswith(surv.CLINICAL_FEATURE_PREFIX):
        return "clinical"
    if feature.startswith(surv.ENGINEERED_FEATURE_PREFIX):
        return "engineered"
    return "radiomics"


def feature_modality(feature: str) -> str:
    if feature.startswith(surv.CLINICAL_FEATURE_PREFIX):
        return "clinical"
    if feature.startswith(surv.ENGINEERED_FEATURE_PREFIX):
        return "engineered"
    return feature.split("_", 1)[0]


def readable_direction(value: float) -> str:
    return "higher value -> higher predicted progression risk" if value > 0 else "higher value -> lower predicted progression risk"


def load_inputs(result_dir: Path) -> tuple[dict[str, object], pd.DataFrame, set[str]]:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    features = pd.read_csv(result_dir / "radiomics_features.csv")
    test_patients = {
        line.strip()
        for line in (result_dir / "test_patients.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return summary, features, test_patients


def fit_final_model(
    train_df: pd.DataFrame,
    summary: dict[str, object],
    seed: int,
) -> tuple[object, surv.PreprocessorState, pd.DataFrame]:
    best_model = summary["best_model"]
    model_name = str(best_model["model"])
    if model_name != "logreg":
        raise ValueError(f"Expected a logistic model, found {model_name!r}")
    columns = list(best_model["columns"])
    params = dict(best_model["params"])
    variance_threshold = float(params.pop("variance_threshold"))
    corr_threshold = float(params.pop("corr_threshold"))
    X_train = train_df[columns]
    X_proc, state = surv.fit_preprocessor(
        X_train,
        variance_threshold=variance_threshold,
        corr_threshold=corr_threshold,
        scale=surv.model_uses_scaling(model_name),
    )
    model = surv.build_model(model_name, params, seed=seed, lightgbm_device="cpu")
    model.fit(X_proc, train_df["label"])
    return model, state, X_proc


def coefficient_table(model, X_proc: pd.DataFrame) -> pd.DataFrame:
    coefs = pd.Series(model.coef_.ravel(), index=X_proc.columns, name="coefficient")
    table = coefs.rename_axis("feature").reset_index()
    table["abs_coefficient"] = table["coefficient"].abs()
    table["odds_ratio_per_sd"] = np.exp(table["coefficient"])
    table["feature_group"] = table["feature"].map(feature_group)
    table["modality"] = table["feature"].map(feature_modality)
    table["direction"] = table["coefficient"].map(readable_direction)
    return table.sort_values("abs_coefficient", ascending=False, kind="stable").reset_index(drop=True)


def stability_table(
    train_df: pd.DataFrame,
    summary: dict[str, object],
    final_columns: list[str],
    cv_folds: int,
    seed: int,
) -> pd.DataFrame:
    best_model = summary["best_model"]
    columns = list(best_model["columns"])
    params = dict(best_model["params"])
    variance_threshold = float(params.pop("variance_threshold"))
    corr_threshold = float(params.pop("corr_threshold"))
    splitter = surv.cv_splitter(
        train_df["label"].to_numpy(),
        train_df["patient_id"].to_numpy(),
        n_splits=cv_folds,
        seed=seed,
        group_aware=True,
    )

    fold_series: list[pd.Series] = []
    for fold_index, (train_idx, val_idx) in enumerate(
        splitter.split(train_df[columns], train_df["label"], train_df["patient_id"]),
        start=1,
    ):
        del val_idx
        fold_train = train_df.iloc[train_idx]
        X_fold, fold_state = surv.fit_preprocessor(
            fold_train[columns],
            variance_threshold=variance_threshold,
            corr_threshold=corr_threshold,
            scale=surv.model_uses_scaling("logreg"),
        )
        model = surv.build_model("logreg", params, seed=seed, lightgbm_device="cpu")
        model.fit(X_fold, fold_train["label"])
        series = pd.Series(model.coef_.ravel(), index=X_fold.columns, name=f"fold_{fold_index}")
        fold_series.append(series.reindex(final_columns).fillna(0.0))

    coef_df = pd.concat(fold_series, axis=1).fillna(0.0)
    stability = pd.DataFrame({"feature": final_columns})
    stability["mean_coefficient"] = coef_df.mean(axis=1).to_numpy()
    stability["std_coefficient"] = coef_df.std(axis=1, ddof=0).to_numpy()
    stability["positive_fold_fraction"] = (coef_df > 0).mean(axis=1).to_numpy()
    stability["negative_fold_fraction"] = (coef_df < 0).mean(axis=1).to_numpy()
    stability["nonzero_fold_fraction"] = (coef_df != 0).mean(axis=1).to_numpy()
    stability["sign_consistency"] = np.maximum(
        stability["positive_fold_fraction"], stability["negative_fold_fraction"]
    )
    stability["feature_group"] = stability["feature"].map(feature_group)
    stability["modality"] = stability["feature"].map(feature_modality)
    return stability.sort_values(
        ["sign_consistency", "nonzero_fold_fraction", "feature"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def local_contributions(
    model,
    state: surv.PreprocessorState,
    test_df: pd.DataFrame,
    selected_columns: list[str],
) -> pd.DataFrame:
    X_test = surv.transform_preprocessor(test_df[selected_columns], state)
    coef = pd.Series(model.coef_.ravel(), index=X_test.columns)
    rows: list[dict[str, object]] = []
    for index, row in X_test.iterrows():
        contributions = (row * coef).sort_values(key=np.abs, ascending=False)
        patient_id = str(test_df.loc[index, "patient_id"])
        timepoint = str(test_df.loc[index, "timepoint"])
        for feature, value in contributions.head(10).items():
            rows.append(
                {
                    "patient_id": patient_id,
                    "timepoint": timepoint,
                    "label": int(test_df.loc[index, "label"]),
                    "feature": feature,
                    "contribution": float(value),
                    "abs_contribution": float(abs(value)),
                    "feature_group": feature_group(feature),
                    "modality": feature_modality(feature),
                }
            )
    return pd.DataFrame(rows)


def build_markdown_summary(
    summary: dict[str, object],
    coefficients: pd.DataFrame,
    stability: pd.DataFrame,
) -> str:
    clinical = coefficients[coefficients["feature_group"] == "clinical"].head(8)
    radiomics_pos = coefficients[coefficients["coefficient"] > 0].head(8)
    radiomics_neg = coefficients[coefficients["coefficient"] < 0].head(8)
    stable = stability.sort_values(
        ["sign_consistency", "nonzero_fold_fraction", "feature"],
        ascending=[False, False, True],
        kind="stable",
    ).head(12)

    lines = [
        "# Logistic Explainability Report",
        "",
        f"- Model: `{summary['best_model']['model']}`",
        f"- Selected feature count after preprocessing: `{len(coefficients)}`",
        f"- Held-out calibrated ROC AUC: `{summary['test_metrics_calibrated']['roc_auc']:.4f}`",
        f"- Held-out calibrated balanced accuracy: `{summary['test_metrics_calibrated']['balanced_accuracy']:.4f}`",
        f"- Held-out calibrated Brier score: `{summary['test_brier_calibrated']:.4f}`",
        "",
        "## Interpretation Notes",
        "",
        "- Coefficients are on the standardized feature scale after the model's train-only preprocessing.",
        "- Positive coefficients indicate higher predicted 120-day progression risk as the feature increases.",
        "- Negative coefficients indicate lower predicted 120-day progression risk as the feature increases.",
        "- Clinical one-hot features use dataset-coded category values such as `__0`, `__1`, or `__2`.",
        "",
        "## Strongest Clinical Terms",
        "",
    ]

    if clinical.empty:
        lines.append("- No clinical terms survived into the final fitted model.")
    else:
        for row in clinical.itertuples(index=False):
            lines.append(
                f"- `{row.feature}`: coef `{row.coefficient:.3f}`, OR/SD `{row.odds_ratio_per_sd:.3f}`, {row.direction}"
            )

    lines.extend(
        [
            "",
            "## Strongest Positive Terms",
            "",
        ]
    )
    for row in radiomics_pos.itertuples(index=False):
        lines.append(
            f"- `{row.feature}`: coef `{row.coefficient:.3f}`, OR/SD `{row.odds_ratio_per_sd:.3f}`, {row.feature_group}"
        )

    lines.extend(
        [
            "",
            "## Strongest Negative Terms",
            "",
        ]
    )
    for row in radiomics_neg.itertuples(index=False):
        lines.append(
            f"- `{row.feature}`: coef `{row.coefficient:.3f}`, OR/SD `{row.odds_ratio_per_sd:.3f}`, {row.feature_group}"
        )

    lines.extend(
        [
            "",
            "## Most Stable Terms Across CV Folds",
            "",
        ]
    )
    for row in stable.itertuples(index=False):
        lines.append(
            f"- `{row.feature}`: sign consistency `{row.sign_consistency:.2f}`, nonzero fold fraction `{row.nonzero_fold_fraction:.2f}`, mean coef `{row.mean_coefficient:.3f}`"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else result_dir / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, features, test_patients = load_inputs(result_dir)
    features["split"] = np.where(features["patient_id"].astype(str).isin(test_patients), "test", "train")
    train_df = features[features["split"] == "train"].copy()
    test_df = features[features["split"] == "test"].copy()

    model, state, X_proc = fit_final_model(train_df, summary, seed=args.seed)
    coefficients = coefficient_table(model, X_proc)
    stability = stability_table(
        train_df=train_df,
        summary=summary,
        final_columns=X_proc.columns.tolist(),
        cv_folds=args.cv_folds,
        seed=args.seed,
    )
    top_contrib = local_contributions(
        model=model,
        state=state,
        test_df=test_df,
        selected_columns=list(summary["best_model"]["columns"]),
    )

    coefficients.to_csv(output_dir / "coefficients.csv", index=False)
    stability.to_csv(output_dir / "coefficient_stability.csv", index=False)
    top_contrib.to_csv(output_dir / "test_case_top_contributions.csv", index=False)
    (output_dir / "summary.md").write_text(
        build_markdown_summary(summary=summary, coefficients=coefficients, stability=stability),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
