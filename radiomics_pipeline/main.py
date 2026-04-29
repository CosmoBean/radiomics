"""Top-level CLI for prep and model training."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run radiomics data prep and model training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep_parser = subparsers.add_parser("prep-data", help="Build manifests and processed inputs.")
    prep_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("PKG-MU-Glioma-Post/MU-Glioma-Post"),
        help="Root directory containing the raw patient folders.",
    )
    prep_parser.add_argument(
        "--clinical-xlsx",
        type=Path,
        default=Path("PKG-MU-Glioma-Post/MU-Glioma-Post_ClinicalData-July2025.xlsx"),
        help="Clinical workbook used to build the experiment index.",
    )
    prep_parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("processed"),
        help="Root directory for processed outputs.",
    )
    prep_parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("metadata"),
        help="Directory for audit outputs and split files.",
    )

    train_parser = subparsers.add_parser("train", help="Run the calibrated forward hybrid training flow.")
    train_parser.add_argument(
        "--experiment-index",
        type=Path,
        default=Path("processed/manifests/experiment_index.csv"),
        help="Merged experiment index CSV.",
    )
    train_parser.add_argument(
        "--radiomics-yaml",
        type=Path,
        default=Path("configs/postoperative_progression_surveillance_radiomics.yaml"),
        help="PyRadiomics settings file.",
    )
    train_parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/calibrated_forward_hybrid"),
        help="Directory for training outputs.",
    )
    train_parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/calibrated"),
        help="Directory for the exported calibrated bundle.",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=62,
        help="Random seed for the preferred calibrated run.",
    )
    train_parser.add_argument(
        "--models",
        type=str,
        default="lightgbm,logreg,rf,svm",
        help="Comma-separated model families to search.",
    )
    train_parser.add_argument(
        "--modalities",
        type=str,
        default="t1c,flair",
        help="Comma-separated imaging modalities.",
    )
    train_parser.add_argument(
        "--clinical-feature-set",
        type=str,
        default="hybrid_basic",
        help="Clinical feature block to merge into training.",
    )
    train_parser.add_argument(
        "--feature-subsets",
        type=str,
        default="16,24,32,48",
        help="Comma-separated ranked feature subset sizes.",
    )
    train_parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="Optuna trials per model/subset.",
    )
    train_parser.add_argument(
        "--ranking-folds",
        type=int,
        default=5,
        help="Ranking folds.",
    )
    train_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds.",
    )
    train_parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=20,
        help="Permutation repeats for feature ranking.",
    )
    train_parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=300,
        help="Bootstrap iterations for the ROC interval.",
    )
    train_parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel workers for feature extraction.",
    )
    return parser


def run_prep_data(args: argparse.Namespace) -> None:
    from radiomics_pipeline.workflows import audit, build_index, preprocess, split_patients

    audit.main(
        [
            "--dataset-root",
            str(args.dataset_root),
            "--output-dir",
            str(args.metadata_dir),
        ]
    )
    split_patients.main(
        [
            "--summary-csv",
            str(args.metadata_dir / "timepoint_summary.csv"),
            "--output-dir",
            str(args.metadata_dir),
        ]
    )
    preprocess.main(
        [
            "--dataset-root",
            str(args.dataset_root),
            "--manifest-csv",
            str(args.metadata_dir / "manifest.csv"),
            "--summary-csv",
            str(args.metadata_dir / "timepoint_summary.csv"),
            "--splits-csv",
            str(args.metadata_dir / "splits.csv"),
            "--output-root",
            str(args.processed_root),
        ]
    )
    build_index.main(
        [
            "--clinical-xlsx",
            str(args.clinical_xlsx),
            "--processed-root",
            str(args.processed_root),
            "--output-dir",
            str(args.processed_root / "manifests"),
        ]
    )


def run_train(args: argparse.Namespace) -> None:
    from radiomics_pipeline.workflows import export_calibrated, train

    train.main(
        [
            "--experiment-index",
            str(args.experiment_index),
            "--radiomics-yaml",
            str(args.radiomics_yaml),
            "--output-dir",
            str(args.result_dir),
            "--label-mode",
            "within_window",
            "--progression-window-days",
            "120",
            "--pre-progression-only",
            "--exclude-after-late-treatment",
            "--models",
            args.models,
            "--modalities",
            args.modalities,
            "--clinical-feature-set",
            args.clinical_feature_set,
            "--feature-subsets",
            args.feature_subsets,
            "--n-trials",
            str(args.n_trials),
            "--ranking-folds",
            str(args.ranking_folds),
            "--cv-folds",
            str(args.cv_folds),
            "--permutation-repeats",
            str(args.permutation_repeats),
            "--bootstrap-iterations",
            str(args.bootstrap_iterations),
            "--max-workers",
            str(args.max_workers),
            "--progress-bar",
            "off",
            "--seed",
            str(args.seed),
        ]
    )
    export_calibrated.main(
        [
            "--result-dir",
            str(args.result_dir),
            "--output-dir",
            str(args.model_dir),
            "--seed",
            str(args.seed),
            "--cv-folds",
            str(args.cv_folds),
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prep-data":
        run_prep_data(args)
        return
    if args.command == "train":
        run_train(args)
        return
    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
