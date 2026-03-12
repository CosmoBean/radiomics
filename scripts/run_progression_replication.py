#!/usr/bin/env python3
"""Replicate a progression-focused radiomics pipeline with probability calibration."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
import warnings

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import optuna
import pandas as pd
from radiomics import featureextractor, logger as radiomics_logger
import shap
import SimpleITK as sitk
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier


MODALITY_PATH_COLUMNS = {
    "t1": "native_t1_path",
    "t1c": "native_t1c_path",
    "t2": "native_t2_path",
    "flair": "native_flair_path",
}

METADATA_COLUMNS = {
    "patient_id",
    "timepoint",
    "timepoint_number",
    "days_from_diagnosis_to_mri",
    "target",
    "split",
    "lesion_voxel_count",
}

_WORKER_ROOT: Path | None = None
_WORKER_LABELS: tuple[int, ...] = ()
_WORKER_EXTRACTOR: featureextractor.RadiomicsFeatureExtractor | None = None


@dataclass
class PreprocessorBundle:
    input_features: list[str]
    imputer: SimpleImputer
    variance_selector: VarianceThreshold
    variance_features: list[str]
    correlation_features: list[str]
    scaler: StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a progression-focused radiomics replication on MU-Glioma-Post."
    )
    parser.add_argument(
        "--experiment-index",
        type=Path,
        default=Path("processed/manifests/experiment_index.csv"),
        help="Merged imaging and clinical index.",
    )
    parser.add_argument(
        "--params-yaml",
        type=Path,
        default=Path("scripts/radiomics_progression_params.yaml"),
        help="PyRadiomics settings file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/progression_replication"),
        help="Directory for extracted features, model artifacts, and reports.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for radiomics extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260311,
        help="Global random seed.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Recompute radiomics features even if cached outputs exist.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract radiomics features and stop before modeling.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--subset-sizes",
        type=str,
        default="512,256,128,64",
        help="Comma-separated ranked feature subset sizes to evaluate.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=400,
        help="Optuna trials per subset.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Group-aware CV folds for tuning and OOF predictions.",
    )
    parser.add_argument(
        "--rank-folds",
        type=int,
        default=10,
        help="Stratified folds for consensus ranking.",
    )
    parser.add_argument(
        "--rank-perm-repeats",
        type=int,
        default=100,
        help="Permutation-importance repeats per ranking fold.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Bootstrap iterations for ROC AUC confidence intervals.",
    )
    parser.add_argument(
        "--target-test-patients",
        type=int,
        default=30,
        help="Desired held-out patient count.",
    )
    parser.add_argument(
        "--target-test-rows",
        type=int,
        default=96,
        help="Desired held-out timepoint count.",
    )
    parser.add_argument(
        "--target-test-negative-rows",
        type=int,
        default=43,
        help="Desired held-out negative-class timepoint count.",
    )
    parser.add_argument(
        "--lesion-labels",
        type=str,
        default="1,2,3",
        help="Comma-separated segmentation labels included in the tumor-union mask.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP explainability outputs.",
    )
    parser.add_argument(
        "--shap-max-samples",
        type=int,
        default=128,
        help="Maximum evaluation samples for SHAP calculations.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path.cwd().resolve()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def repo_relative(path: Path, root: Path) -> str:
    absolute = path if path.is_absolute() else root / path
    return absolute.relative_to(root).as_posix()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_usable_cohort(index_csv: Path, max_cases: int | None) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    usable = df[(df["roi_status"] == "written") & df["clinical_progression"].isin([0, 1])].copy()
    usable["target"] = usable["clinical_progression"].astype(int)
    usable = usable[usable["source_mask_path"].astype(str).ne("")]
    for column in MODALITY_PATH_COLUMNS.values():
        usable = usable[usable[column].astype(str).ne("")]
    usable = usable.sort_values(["patient_id", "timepoint_number"], kind="stable").reset_index(drop=True)
    if max_cases is not None:
        usable = usable.head(max_cases).copy()
    return usable


def exact_subset_by_count_and_sum(
    items: list[tuple[str, int]],
    choose_k: int,
    target_sum: int,
) -> tuple[str, ...] | None:
    if choose_k < 0 or target_sum < 0:
        return None
    dp: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
    for patient_id, row_count in items:
        current = list(dp.items())
        updates: dict[tuple[int, int], tuple[str, ...]] = {}
        for (picked, total_rows), selected in current:
            next_picked = picked + 1
            next_rows = total_rows + row_count
            if next_picked > choose_k or next_rows > target_sum:
                continue
            key = (next_picked, next_rows)
            candidate = tuple(sorted((*selected, patient_id)))
            existing = dp.get(key) or updates.get(key)
            if existing is None or candidate < existing:
                updates[key] = candidate
        dp.update(updates)
    return dp.get((choose_k, target_sum))


def choose_test_patients(
    cohort: pd.DataFrame,
    target_patients: int,
    target_rows: int,
    target_negative_rows: int,
) -> tuple[list[str], dict[str, int]]:
    patient_table = (
        cohort.groupby("patient_id")
        .agg(row_count=("timepoint", "size"), target=("target", "first"))
        .reset_index()
    )
    positives = [
        (str(row.patient_id), int(row.row_count))
        for row in patient_table.itertuples(index=False)
        if int(row.target) == 1
    ]
    negatives = [
        (str(row.patient_id), int(row.row_count))
        for row in patient_table.itertuples(index=False)
        if int(row.target) == 0
    ]
    positives = sorted(positives)
    negatives = sorted(negatives)

    target_positive_rows = target_rows - target_negative_rows
    patient_negative_rate = float((patient_table["target"] == 0).mean())
    expected_negative_patients = int(round(target_patients * patient_negative_rate))

    candidates: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []
    for negative_patients in range(max(0, target_patients - len(positives)), min(target_patients, len(negatives)) + 1):
        positive_patients = target_patients - negative_patients
        negative_subset = exact_subset_by_count_and_sum(
            negatives,
            choose_k=negative_patients,
            target_sum=target_negative_rows,
        )
        if negative_subset is None:
            continue
        positive_subset = exact_subset_by_count_and_sum(
            positives,
            choose_k=positive_patients,
            target_sum=target_positive_rows,
        )
        if positive_subset is None:
            continue
        penalty = abs(negative_patients - expected_negative_patients)
        candidates.append((penalty, negative_subset, positive_subset))

    if not candidates:
        raise RuntimeError(
            "Could not find a patient-held-out subset matching the requested test-set targets."
        )

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, negative_subset, positive_subset = candidates[0]
    selected = sorted((*negative_subset, *positive_subset))
    selected_set = set(selected)
    held_out = cohort[cohort["patient_id"].isin(selected_set)].copy()
    counts = {
        "test_patients": int(len(selected)),
        "test_rows": int(len(held_out)),
        "test_negative_rows": int((held_out["target"] == 0).sum()),
        "test_positive_rows": int((held_out["target"] == 1).sum()),
        "test_negative_patients": int(len(negative_subset)),
        "test_positive_patients": int(len(positive_subset)),
    }
    return selected, counts


def init_worker(root_str: str, params_yaml_str: str, lesion_labels: tuple[int, ...]) -> None:
    global _WORKER_ROOT, _WORKER_LABELS, _WORKER_EXTRACTOR
    _WORKER_ROOT = Path(root_str)
    _WORKER_LABELS = lesion_labels
    warnings.filterwarnings("ignore")
    radiomics_logger.setLevel(logging.ERROR)
    _WORKER_EXTRACTOR = featureextractor.RadiomicsFeatureExtractor(
        str(params_yaml_str),
        enableCExtensions=True,
    )
    _WORKER_EXTRACTOR.settings["label"] = 1


def sitk_mask_from_labels(mask_image: sitk.Image, lesion_labels: tuple[int, ...]) -> sitk.Image:
    mask_array = sitk.GetArrayFromImage(mask_image)
    lesion_array = np.isin(mask_array, lesion_labels).astype(np.uint8)
    lesion_image = sitk.GetImageFromArray(lesion_array)
    lesion_image.CopyInformation(mask_image)
    return lesion_image


def apply_n4(image: sitk.Image, lesion_mask: sitk.Image) -> sitk.Image:
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    return corrector.Execute(sitk.Cast(image, sitk.sitkFloat32), sitk.Cast(lesion_mask, sitk.sitkUInt8))


def zscore_within_mask(image: sitk.Image, lesion_mask: sitk.Image) -> sitk.Image:
    image_array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    mask_array = sitk.GetArrayFromImage(lesion_mask) > 0
    values = image_array[mask_array]
    normalized = np.zeros_like(image_array, dtype=np.float32)
    if values.size == 0:
        output = sitk.GetImageFromArray(normalized)
        output.CopyInformation(image)
        return output
    mean_value = float(values.mean())
    std_value = float(values.std())
    denom = std_value if std_value > 1e-8 else 1.0
    normalized = (image_array - mean_value) / denom
    normalized[~np.isfinite(normalized)] = 0.0
    output = sitk.GetImageFromArray(normalized.astype(np.float32, copy=False))
    output.CopyInformation(image)
    return output


def extract_case_features(row: dict[str, object]) -> dict[str, object]:
    if _WORKER_ROOT is None or _WORKER_EXTRACTOR is None:
        raise RuntimeError("Radiomics worker was not initialized.")

    record: dict[str, object] = {
        "patient_id": str(row["patient_id"]),
        "timepoint": str(row["timepoint"]),
        "timepoint_number": int(row["timepoint_number"]),
        "days_from_diagnosis_to_mri": float(row["days_from_diagnosis_to_mri"])
        if not pd.isna(row["days_from_diagnosis_to_mri"])
        else np.nan,
        "target": int(row["target"]),
    }

    mask_path = _WORKER_ROOT / str(row["source_mask_path"])
    mask_image = sitk.ReadImage(str(mask_path))
    lesion_mask = sitk_mask_from_labels(mask_image, _WORKER_LABELS)
    lesion_voxels = int(sitk.GetArrayFromImage(lesion_mask).sum())
    if lesion_voxels == 0:
        raise RuntimeError(
            f"No lesion voxels remain after label filtering for {row['patient_id']} {row['timepoint']}"
        )
    record["lesion_voxel_count"] = lesion_voxels

    for modality, path_column in MODALITY_PATH_COLUMNS.items():
        image_path = _WORKER_ROOT / str(row[path_column])
        image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
        corrected = apply_n4(image, lesion_mask)
        normalized = zscore_within_mask(corrected, lesion_mask)
        features = _WORKER_EXTRACTOR.execute(normalized, lesion_mask)
        for name, value in features.items():
            if name.startswith("diagnostics_"):
                continue
            feature_name = name.replace("original_", "")
            record[f"{modality}__{feature_name}"] = float(value)

    return record


def extract_features(
    cohort: pd.DataFrame,
    output_dir: Path,
    params_yaml: Path,
    workers: int,
    lesion_labels: tuple[int, ...],
    force_extract: bool,
) -> pd.DataFrame:
    repo = repo_root()
    features_csv = output_dir / "radiomics_features.csv"

    if features_csv.exists() and not force_extract:
        cached = pd.read_csv(features_csv)
        keep = cohort[["patient_id", "timepoint"]].drop_duplicates()
        features = cached.merge(keep, on=["patient_id", "timepoint"], how="inner")
        return features.sort_values(["patient_id", "timepoint_number"], kind="stable").reset_index(drop=True)

    rows = cohort.to_dict(orient="records")
    extracted: list[dict[str, object]] = []
    max_workers = max(1, min(workers, len(rows)))
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(str(repo), str(params_yaml.resolve()), lesion_labels),
    ) as executor:
        futures = {executor.submit(extract_case_features, row): row for row in rows}
        total = len(futures)
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                extracted.append(future.result())
            except Exception as exc:  # pragma: no cover - surfaced to caller
                raise RuntimeError(
                    f"Radiomics extraction failed for {row['patient_id']} {row['timepoint']}: {exc}"
                ) from exc
            if index % 10 == 0 or index == total:
                print(f"Extracted radiomics for {index}/{total} timepoints", flush=True)

    features = pd.DataFrame(extracted).sort_values(
        ["patient_id", "timepoint_number"],
        kind="stable",
    )
    features.to_csv(features_csv, index=False)
    return features


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in METADATA_COLUMNS]


def spearman_filter(X_train: pd.DataFrame, threshold: float) -> list[str]:
    corr = X_train.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop: set[str] = set()
    for column in upper.columns:
        if column in drop:
            continue
        correlated = upper.index[upper[column] >= threshold].tolist()
        drop.update(correlated)
    return [column for column in X_train.columns if column not in drop]


def fit_rank_filters(
    X_train: pd.DataFrame,
    variance_threshold: float,
    corr_threshold: float,
) -> tuple[pd.Series, list[str], list[str]]:
    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    medians = pd.Series(imputer.statistics_, index=X_train.columns, dtype=np.float64)
    variance = VarianceThreshold(threshold=variance_threshold)
    variance.fit(imputed)
    variance_columns = imputed.columns[variance.get_support(indices=True)].tolist()
    variance_df = imputed[variance_columns].copy()
    corr_columns = spearman_filter(variance_df, threshold=corr_threshold)
    return medians, variance_columns, corr_columns


def rank_features(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    rank_folds: int,
    perm_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    X_train = train_df[feature_columns]
    y_train = train_df["target"].astype(int)

    medians, variance_columns, corr_columns = fit_rank_filters(
        X_train,
        variance_threshold=1e-8,
        corr_threshold=0.80,
    )
    ranked_input = X_train[corr_columns].fillna(medians.reindex(corr_columns)).copy()

    skf = StratifiedKFold(n_splits=rank_folds, shuffle=True, random_state=seed)
    coefficient_rows: list[np.ndarray] = []
    permutation_rows: list[np.ndarray] = []

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(ranked_input, y_train), start=1):
        X_fold_train = ranked_input.iloc[train_idx]
        X_fold_val = ranked_input.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        penalty="l1",
                        C=1.0,
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=seed + fold_index,
                    ),
                ),
            ]
        )
        pipeline.fit(X_fold_train, y_fold_train)
        coefficient_rows.append(np.abs(pipeline.named_steps["model"].coef_.ravel()))

        perm = permutation_importance(
            pipeline,
            X_fold_val,
            y_fold_val,
            scoring="roc_auc",
            n_repeats=perm_repeats,
            random_state=seed + fold_index,
            n_jobs=1,
        )
        permutation_rows.append(perm.importances_mean)
        print(f"Ranking fold {fold_index}/{rank_folds} complete", flush=True)

    coefficient_array = np.vstack(coefficient_rows)
    permutation_array = np.vstack(permutation_rows)

    ranking = pd.DataFrame(
        {
            "feature": corr_columns,
            "median_abs_l1_coefficient": np.median(coefficient_array, axis=0),
            "median_permutation_importance": np.median(permutation_array, axis=0),
        }
    )
    ranking["l1_rank"] = ranking["median_abs_l1_coefficient"].rank(
        method="min",
        ascending=False,
    )
    ranking["permutation_rank"] = ranking["median_permutation_importance"].rank(
        method="min",
        ascending=False,
    )
    ranking["consensus_rank"] = ranking["l1_rank"] + ranking["permutation_rank"]
    ranking = ranking.sort_values(
        ["consensus_rank", "median_abs_l1_coefficient", "median_permutation_importance", "feature"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    summary = {
        "raw_feature_count": int(len(feature_columns)),
        "post_variance_feature_count": int(len(variance_columns)),
        "post_correlation_feature_count": int(len(corr_columns)),
    }
    return ranking, summary


def fit_preprocessor(
    X_train: pd.DataFrame,
    variance_threshold: float,
    corr_threshold: float,
) -> tuple[PreprocessorBundle, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    imputed_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    variance_selector = VarianceThreshold(threshold=variance_threshold)
    variance_array = variance_selector.fit_transform(imputed_train)
    variance_features = imputed_train.columns[variance_selector.get_support(indices=True)].tolist()
    if not variance_features:
        raise ValueError("Variance filtering removed every feature.")
    variance_df = pd.DataFrame(variance_array, columns=variance_features, index=X_train.index)
    corr_features = spearman_filter(variance_df, threshold=corr_threshold)
    if not corr_features:
        raise ValueError("Correlation filtering removed every feature.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(variance_df[corr_features])
    bundle = PreprocessorBundle(
        input_features=list(X_train.columns),
        imputer=imputer,
        variance_selector=variance_selector,
        variance_features=variance_features,
        correlation_features=corr_features,
        scaler=scaler,
    )
    return bundle, scaled


def transform_with_preprocessor(bundle: PreprocessorBundle, X: pd.DataFrame) -> np.ndarray:
    imputed = pd.DataFrame(
        bundle.imputer.transform(X[bundle.input_features]),
        columns=bundle.input_features,
        index=X.index,
    )
    variance_df = pd.DataFrame(
        bundle.variance_selector.transform(imputed),
        columns=bundle.variance_features,
        index=X.index,
    )
    scaled = bundle.scaler.transform(variance_df[bundle.correlation_features])
    return scaled


def build_lgbm(params: dict[str, float | int], seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        verbosity=-1,
        deterministic=True,
        force_row_wise=True,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        learning_rate=float(params["learning_rate"]),
        n_estimators=int(params["n_estimators"]),
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
    )


def suggest_trial_params(trial: optuna.trial.Trial) -> dict[str, float | int]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "variance_threshold": trial.suggest_float("variance_threshold", 1e-12, 1e-4, log=True),
        "corr_threshold": trial.suggest_float("corr_threshold", 0.5, 0.95),
    }


def tune_subset(
    train_df: pd.DataFrame,
    ranked_features: list[str],
    subset_size: int,
    optuna_trials: int,
    cv_folds: int,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    subset_features = ranked_features[:subset_size]
    X_subset = train_df[subset_features].copy()
    y = train_df["target"].astype(int).to_numpy()
    groups = train_df["patient_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_trial_params(trial)
        fold_aucs: list[float] = []
        fold_kept_features: list[int] = []
        for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X_subset, y, groups), start=1):
            X_fold_train = X_subset.iloc[train_idx]
            X_fold_val = X_subset.iloc[val_idx]
            y_fold_train = y[train_idx]
            y_fold_val = y[val_idx]
            try:
                bundle, X_train_processed = fit_preprocessor(
                    X_fold_train,
                    variance_threshold=float(params["variance_threshold"]),
                    corr_threshold=float(params["corr_threshold"]),
                )
            except ValueError:
                return 0.0
            X_val_processed = transform_with_preprocessor(bundle, X_fold_val)
            model = build_lgbm(params, seed=seed + trial.number + fold_index)
            model.fit(X_train_processed, y_fold_train)
            probabilities = model.predict_proba(X_val_processed)[:, 1]
            auc = roc_auc_score(y_fold_val, probabilities)
            fold_aucs.append(float(auc))
            fold_kept_features.append(len(bundle.correlation_features))

        trial.set_user_attr("mean_fold_auc", float(np.mean(fold_aucs)))
        trial.set_user_attr("kept_feature_count", int(np.mean(fold_kept_features)))
        return float(np.max(fold_aucs))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)

    trial_rows = []
    for trial in study.trials:
        row = {"trial_number": trial.number, "objective_best_fold_auc": trial.value}
        row.update(trial.params)
        row["mean_fold_auc"] = trial.user_attrs.get("mean_fold_auc")
        row["kept_feature_count"] = trial.user_attrs.get("kept_feature_count")
        trial_rows.append(row)
    trials_df = pd.DataFrame(trial_rows).sort_values(
        ["objective_best_fold_auc", "mean_fold_auc"],
        ascending=[False, False],
        kind="stable",
    )
    trials_df.to_csv(output_dir / f"lightgbm_trials_top{subset_size}.csv", index=False)

    best = study.best_trial
    summary = {
        "subset_size": int(subset_size),
        "objective_best_fold_auc": float(best.value),
        "mean_fold_auc": float(best.user_attrs.get("mean_fold_auc", np.nan)),
        "kept_feature_count": int(best.user_attrs.get("kept_feature_count", 0)),
        "params": dict(best.params),
    }
    return summary, trials_df


def choose_threshold_min_errors(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, int | float]]:
    thresholds = np.unique(np.quantile(probabilities, np.linspace(0.01, 0.99, 199)))
    best_threshold = 0.5
    best_score = None
    best_stats: dict[str, int | float] = {}
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        errors = int(fp + fn)
        candidate = (errors, -tp, fp)
        if best_score is None or candidate < best_score:
            best_score = candidate
            best_threshold = float(threshold)
            best_stats = {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "errors": errors,
            }
    return best_threshold, best_stats


def fit_final_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ranked_features: list[str],
    subset_size: int,
    best_params: dict[str, object],
    cv_folds: int,
    seed: int,
) -> dict[str, object]:
    subset_features = ranked_features[:subset_size]
    X_train = train_df[subset_features].copy()
    X_test = test_df[subset_features].copy()
    y_train = train_df["target"].astype(int).to_numpy()
    y_test = test_df["target"].astype(int).to_numpy()
    groups = train_df["patient_id"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_raw = np.full(len(train_df), np.nan, dtype=np.float64)
    fold_aucs: list[float] = []

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_train, groups), start=1):
        bundle, X_fold_train = fit_preprocessor(
            X_train.iloc[train_idx],
            variance_threshold=float(best_params["variance_threshold"]),
            corr_threshold=float(best_params["corr_threshold"]),
        )
        X_fold_val = transform_with_preprocessor(bundle, X_train.iloc[val_idx])
        model = build_lgbm(best_params, seed=seed + fold_index)
        model.fit(X_fold_train, y_train[train_idx])
        probabilities = model.predict_proba(X_fold_val)[:, 1]
        oof_raw[val_idx] = probabilities
        fold_aucs.append(float(roc_auc_score(y_train[val_idx], probabilities)))

    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(oof_raw.reshape(-1, 1), y_train)
    oof_calibrated = calibrator.predict_proba(oof_raw.reshape(-1, 1))[:, 1]
    threshold, threshold_stats = choose_threshold_min_errors(y_train, oof_calibrated)

    full_bundle, X_train_processed = fit_preprocessor(
        X_train,
        variance_threshold=float(best_params["variance_threshold"]),
        corr_threshold=float(best_params["corr_threshold"]),
    )
    final_model = build_lgbm(best_params, seed=seed)
    final_model.fit(X_train_processed, y_train)

    X_test_processed = transform_with_preprocessor(full_bundle, X_test)
    test_raw = final_model.predict_proba(X_test_processed)[:, 1]
    test_calibrated = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
    test_predictions = (test_calibrated >= threshold).astype(int)

    return {
        "subset_features": subset_features,
        "preprocessor": full_bundle,
        "model": final_model,
        "calibrator": calibrator,
        "threshold": threshold,
        "threshold_stats": threshold_stats,
        "oof_raw": oof_raw,
        "oof_calibrated": oof_calibrated,
        "fold_aucs": fold_aucs,
        "test_raw": test_raw,
        "test_calibrated": test_calibrated,
        "test_predictions": test_predictions,
        "test_processed": X_test_processed,
        "train_processed": X_train_processed,
        "train_target": y_train,
        "test_target": y_test,
    }


def classification_metrics(y_true: np.ndarray, raw_probabilities: np.ndarray, calibrated_probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, calibrated_probabilities)),
        "brier_raw": float(brier_score_loss(y_true, raw_probabilities)),
        "brier_calibrated": float(brier_score_loss(y_true, calibrated_probabilities)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }


def bootstrap_auc_ci(y_true: np.ndarray, probabilities: np.ndarray, iterations: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    attempts = 0
    max_attempts = iterations * 10
    while len(aucs) < iterations and attempts < max_attempts:
        attempts += 1
        sample_idx = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[sample_idx]
        if np.unique(sample_y).size < 2:
            continue
        aucs.append(float(roc_auc_score(sample_y, probabilities[sample_idx])))
    if not aucs:
        return [float("nan"), float("nan")]
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def decision_curve(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    thresholds = np.linspace(0.01, 0.99, 99)
    prevalence = float(np.mean(y_true == 1))
    rows = []
    n = float(len(y_true))
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        tp = float(((predicted == 1) & (y_true == 1)).sum())
        fp = float(((predicted == 1) & (y_true == 0)).sum())
        weight = threshold / (1.0 - threshold)
        rows.append(
            {
                "threshold_probability": float(threshold),
                "net_benefit_model": tp / n - fp / n * weight,
                "net_benefit_treat_all": prevalence - (1.0 - prevalence) * weight,
                "net_benefit_treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def shap_summary(
    model: LGBMClassifier,
    X_processed: np.ndarray,
    feature_names: list[str],
    max_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(X_processed) > max_samples:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(X_processed), size=max_samples, replace=False))
        X_input = X_processed[selected]
    else:
        X_input = X_processed
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)
    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values[-1], dtype=np.float64)
    else:
        shap_array = np.asarray(shap_values, dtype=np.float64)
    mean_abs = np.abs(shap_array).mean(axis=0)
    feature_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "modality": [name.split("__", 1)[0] for name in feature_names],
        }
    ).sort_values("mean_abs_shap", ascending=False, kind="stable")
    modality_df = (
        feature_df.groupby("modality", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False, kind="stable")
    )
    modality_df["fraction_of_total"] = modality_df["mean_abs_shap"] / modality_df["mean_abs_shap"].sum()
    return feature_df, modality_df


def save_plots(
    output_dir: Path,
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    dca_df: pd.DataFrame,
    shap_df: pd.DataFrame | None,
) -> None:
    fpr_raw, tpr_raw, _ = roc_curve(y_true, raw_probabilities)
    fpr_cal, tpr_cal, _ = roc_curve(y_true, calibrated_probabilities)
    auc_raw = roc_auc_score(y_true, raw_probabilities)
    auc_cal = roc_auc_score(y_true, calibrated_probabilities)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_raw, tpr_raw, label=f"Raw (AUC={auc_raw:.3f})")
    plt.plot(fpr_cal, tpr_cal, label=f"Platt (AUC={auc_cal:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Progression Replication ROC")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6.5, 5))
    plt.plot(
        dca_df["threshold_probability"],
        dca_df["net_benefit_model"],
        label="Model",
        linewidth=2,
    )
    plt.plot(
        dca_df["threshold_probability"],
        dca_df["net_benefit_treat_all"],
        label="Treat-all",
        linestyle="--",
    )
    plt.plot(
        dca_df["threshold_probability"],
        dca_df["net_benefit_treat_none"],
        label="Treat-none",
        linestyle="-.",
    )
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "decision_curve.png", dpi=200)
    plt.close()

    if shap_df is not None and not shap_df.empty:
        top_shap = shap_df.head(20).sort_values("mean_abs_shap", ascending=True, kind="stable")
        plt.figure(figsize=(8, 6))
        plt.barh(top_shap["feature"], top_shap["mean_abs_shap"])
        plt.xlabel("Mean |SHAP|")
        plt.title("Top SHAP Features")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_top20.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")
    radiomics_logger.setLevel(logging.ERROR)

    start = time.time()
    repo = repo_root()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lesion_labels = tuple(parse_int_list(args.lesion_labels))
    subset_sizes = [int(value) for value in parse_int_list(args.subset_sizes)]

    cohort = load_usable_cohort(args.experiment_index.resolve(), args.max_cases)
    test_patients, split_counts = choose_test_patients(
        cohort,
        target_patients=args.target_test_patients,
        target_rows=args.target_test_rows,
        target_negative_rows=args.target_test_negative_rows,
    )
    cohort = cohort.copy()
    cohort["split"] = np.where(cohort["patient_id"].isin(test_patients), "test", "traincv")
    cohort.to_csv(output_dir / "cohort_with_replication_split.csv", index=False)
    (output_dir / "test_patients.txt").write_text("\n".join(test_patients) + "\n", encoding="utf-8")

    features = extract_features(
        cohort=cohort,
        output_dir=output_dir,
        params_yaml=args.params_yaml,
        workers=args.workers,
        lesion_labels=lesion_labels,
        force_extract=args.force_extract,
    )
    features = features.merge(
        cohort[["patient_id", "timepoint", "split"]],
        on=["patient_id", "timepoint"],
        how="left",
        suffixes=("", "_cohort"),
    )
    if "split_cohort" in features.columns:
        features["split"] = features["split_cohort"]
        features = features.drop(columns=["split_cohort"])
    features = features.sort_values(["patient_id", "timepoint_number"], kind="stable").reset_index(drop=True)
    features.to_csv(output_dir / "radiomics_features.csv", index=False)

    extraction_summary = {
        "elapsed_seconds": round(time.time() - start, 3),
        "usable_rows": int(len(cohort)),
        "usable_patients": int(cohort["patient_id"].nunique()),
        "lesion_labels": list(lesion_labels),
        "feature_columns": int(len(get_feature_columns(features))),
        **split_counts,
    }
    write_json(output_dir / "extraction_summary.json", extraction_summary)
    if args.extract_only:
        print(json.dumps(extraction_summary, indent=2, sort_keys=True))
        return

    train_df = features[features["split"] == "traincv"].copy().reset_index(drop=True)
    test_df = features[features["split"] == "test"].copy().reset_index(drop=True)
    feature_columns = get_feature_columns(features)

    ranking, ranking_summary = rank_features(
        train_df=train_df,
        feature_columns=feature_columns,
        rank_folds=args.rank_folds,
        perm_repeats=args.rank_perm_repeats,
        seed=args.seed,
    )
    ranking.to_csv(output_dir / "feature_ranking.csv", index=False)
    ranked_features = ranking["feature"].tolist()
    effective_subset_sizes = []
    for size in subset_sizes:
        effective = min(size, len(ranked_features))
        if effective not in effective_subset_sizes:
            effective_subset_sizes.append(effective)

    subset_summaries: list[dict[str, object]] = []
    for subset_size in effective_subset_sizes:
        print(f"Tuning LightGBM for top-{subset_size} features", flush=True)
        subset_summary, _ = tune_subset(
            train_df=train_df,
            ranked_features=ranked_features,
            subset_size=subset_size,
            optuna_trials=args.optuna_trials,
            cv_folds=args.cv_folds,
            seed=args.seed,
            output_dir=output_dir,
        )
        subset_summaries.append(subset_summary)

    subset_df = pd.DataFrame(subset_summaries).sort_values(
        ["subset_size"],
        kind="stable",
    )
    subset_df.to_csv(output_dir / "subset_tuning_summary.csv", index=False)

    preferred_subset = 256 if 256 in effective_subset_sizes else effective_subset_sizes[0]
    final_subset_row = subset_df[subset_df["subset_size"] == preferred_subset].iloc[0].to_dict()
    final_fit = fit_final_model(
        train_df=train_df,
        test_df=test_df,
        ranked_features=ranked_features,
        subset_size=preferred_subset,
        best_params=final_subset_row["params"],
        cv_folds=args.cv_folds,
        seed=args.seed,
    )

    metrics = classification_metrics(
        y_true=final_fit["test_target"],
        raw_probabilities=final_fit["test_raw"],
        calibrated_probabilities=final_fit["test_calibrated"],
        predictions=final_fit["test_predictions"],
    )
    metrics["roc_auc_ci_95"] = bootstrap_auc_ci(
        y_true=final_fit["test_target"],
        probabilities=final_fit["test_calibrated"],
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    metrics["train_oof_auc"] = float(roc_auc_score(final_fit["train_target"], final_fit["oof_raw"]))
    metrics["train_oof_auc_mean_fold"] = float(np.mean(final_fit["fold_aucs"]))
    metrics["threshold"] = float(final_fit["threshold"])
    metrics["threshold_stats"] = final_fit["threshold_stats"]

    prediction_df = test_df[["patient_id", "timepoint", "timepoint_number", "target"]].copy()
    prediction_df["predicted_probability_raw"] = final_fit["test_raw"]
    prediction_df["predicted_probability_calibrated"] = final_fit["test_calibrated"]
    prediction_df["predicted_class"] = final_fit["test_predictions"]
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)

    confusion_df = pd.DataFrame(
        [
            [metrics["confusion_matrix"]["tn"], metrics["confusion_matrix"]["fp"]],
            [metrics["confusion_matrix"]["fn"], metrics["confusion_matrix"]["tp"]],
        ],
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
    confusion_df.to_csv(output_dir / "confusion_matrix.csv")

    dca_df = decision_curve(final_fit["test_target"], final_fit["test_calibrated"])
    dca_df.to_csv(output_dir / "decision_curve.csv", index=False)

    shap_feature_df: pd.DataFrame | None = None
    modality_df: pd.DataFrame | None = None
    if not args.skip_shap:
        shap_feature_df, modality_df = shap_summary(
            model=final_fit["model"],
            X_processed=final_fit["test_processed"],
            feature_names=final_fit["preprocessor"].correlation_features,
            max_samples=args.shap_max_samples,
            seed=args.seed,
        )
        shap_feature_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)
        modality_df.to_csv(output_dir / "shap_modality_importance.csv", index=False)

    save_plots(
        output_dir=output_dir,
        y_true=final_fit["test_target"],
        raw_probabilities=final_fit["test_raw"],
        calibrated_probabilities=final_fit["test_calibrated"],
        dca_df=dca_df,
        shap_df=shap_feature_df,
    )

    summary = {
        "notes": {
            "plan_counts_reconciled": "The local cohort contains 594 usable masked timepoints across 203 patients; the modified plan's 494-sample figure is not reachable on this export.",
            "reference_repo_mismatch": "The cloned external reference repository is MGMT-focused, not progression-focused postoperative surveillance code.",
        },
        "cohort": {
            "usable_rows": int(len(cohort)),
            "usable_patients": int(cohort["patient_id"].nunique()),
            **split_counts,
        },
        "ranking": ranking_summary,
        "subset_tuning": subset_summaries,
        "final_model": {
            "preferred_subset_size": int(preferred_subset),
            "params": final_subset_row["params"],
        },
        "metrics": metrics,
        "top_ranked_features": ranking.head(20).to_dict(orient="records"),
        "top_shap_features": [] if shap_feature_df is None else shap_feature_df.head(20).to_dict(orient="records"),
        "shap_modality_breakdown": [] if modality_df is None else modality_df.to_dict(orient="records"),
        "elapsed_seconds_total": round(time.time() - start, 3),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
