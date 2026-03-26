#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat the forward hybrid surveillance run across multiple patient-held-out splits."
    )
    parser.add_argument(
        "--feature-table",
        type=Path,
        default=Path("results/radiomics_features.csv"),
        help="Precomputed radiomics feature table to reuse.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/repeated_forward_hybrid_basic"),
        help="Directory for repeated evaluation outputs.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,52,62,72,82",
        help="Comma-separated random seeds for repeated patient-held-out splits.",
    )
    parser.add_argument(
        "--clinical-feature-set",
        type=str,
        default="hybrid_basic",
        help="Clinical feature set to evaluate.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="t1c,flair",
        help="Comma-separated imaging modalities.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="Optuna trials per run.",
    )
    parser.add_argument(
        "--feature-subsets",
        type=str,
        default="16,24,32,48",
        help="Comma-separated subset sizes.",
    )
    parser.add_argument(
        "--ranking-folds",
        type=int,
        default=5,
        help="Ranking folds per run.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="CV folds per run.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=20,
        help="Permutation repeats per run.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=300,
        help="Bootstrap iterations per run.",
    )
    parser.add_argument(
        "--target-test-patients",
        type=int,
        default=30,
        help="Held-out patient count target passed through to the surveillance pipeline.",
    )
    parser.add_argument(
        "--target-test-samples",
        type=int,
        default=96,
        help="Held-out sample count target passed through to the surveillance pipeline.",
    )
    parser.add_argument(
        "--target-test-positives",
        type=int,
        default=53,
        help="Held-out positive count target passed through to the surveillance pipeline.",
    )
    parser.add_argument(
        "--earliest-scan-only",
        action="store_true",
        help="Keep only the earliest usable postoperative scan per patient in each run.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use.",
    )
    return parser.parse_args()


def summarize_runs(run_rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = pd.DataFrame(run_rows)
    numeric_cols = [
        "roc_auc",
        "balanced_accuracy",
        "brier",
        "cv_mean_auc",
        "selected_samples",
        "selected_positives",
        "selected_negatives",
    ]
    summary = {}
    for col in numeric_cols:
        values = metrics[col].astype(float)
        summary[col] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path("scripts/run_postoperative_progression_surveillance.py").resolve()
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    run_rows: list[dict[str, object]] = []

    for seed in seeds:
        run_dir = output_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            str(script_path),
            "--feature-table",
            str(args.feature_table.resolve()),
            "--output-dir",
            str(run_dir),
            "--models",
            "logreg",
            "--modalities",
            args.modalities,
            "--clinical-feature-set",
            args.clinical_feature_set,
            "--label-mode",
            "within_window",
            "--progression-window-days",
            "120",
            "--pre-progression-only",
            "--exclude-after-late-treatment",
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
            "--progress-bar",
            "off",
            "--seed",
            str(seed),
            "--target-test-patients",
            str(args.target_test_patients),
            "--target-test-samples",
            str(args.target_test_samples),
            "--target-test-positives",
            str(args.target_test_positives),
        ]
        if args.earliest_scan_only:
            cmd.append("--earliest-scan-only")
        print("Running:", shlex.join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        split = summary["split_summary"]
        run_rows.append(
            {
                "seed": seed,
                "result_dir": str(run_dir),
                "roc_auc": float(summary["test_metrics_calibrated"]["roc_auc"]),
                "balanced_accuracy": float(summary["test_metrics_calibrated"]["balanced_accuracy"]),
                "brier": float(summary["test_brier_calibrated"]),
                "cv_mean_auc": float(summary["best_model"]["mean_fold_auc"]),
                "subset_size": int(summary["best_model"]["subset_size"]),
                "selected_samples": int(split["selected_samples"]),
                "selected_positives": int(split["selected_positives"]),
                "selected_negatives": int(split["selected_negatives"]),
            }
        )

    run_df = pd.DataFrame(run_rows).sort_values("seed", kind="stable")
    run_df.to_csv(output_dir / "repeated_run_metrics.csv", index=False)

    aggregate = {
        "seeds": seeds,
        "n_runs": len(seeds),
        "clinical_feature_set": args.clinical_feature_set,
        "modalities": args.modalities,
        "earliest_scan_only": bool(args.earliest_scan_only),
        "target_test_patients": int(args.target_test_patients),
        "target_test_samples": int(args.target_test_samples),
        "target_test_positives": int(args.target_test_positives),
        "summary": summarize_runs(run_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Repeated Forward Evaluation",
        "",
        f"- Runs: `{len(seeds)}`",
        f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- Clinical feature set: `{args.clinical_feature_set}`",
        f"- Modalities: `{args.modalities}`",
        f"- Earliest scan only: `{args.earliest_scan_only}`",
        f"- Target held-out patients / samples / positives: `{args.target_test_patients} / {args.target_test_samples} / {args.target_test_positives}`",
        "",
        "## Per-Run ROC AUC",
        "",
    ]
    for row in run_df.itertuples(index=False):
        lines.append(
            f"- seed `{row.seed}`: ROC AUC `{row.roc_auc:.4f}`, balanced accuracy `{row.balanced_accuracy:.4f}`, Brier `{row.brier:.4f}`, held-out `{row.selected_samples}` scans"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- ROC AUC mean ± sd: `{aggregate['summary']['roc_auc']['mean']:.4f} ± {aggregate['summary']['roc_auc']['std']:.4f}`",
            f"- ROC AUC range: `{aggregate['summary']['roc_auc']['min']:.4f}` to `{aggregate['summary']['roc_auc']['max']:.4f}`",
            f"- Balanced accuracy mean ± sd: `{aggregate['summary']['balanced_accuracy']['mean']:.4f} ± {aggregate['summary']['balanced_accuracy']['std']:.4f}`",
            f"- Brier mean ± sd: `{aggregate['summary']['brier']['mean']:.4f} ± {aggregate['summary']['brier']['std']:.4f}`",
            f"- CV mean AUC mean ± sd: `{aggregate['summary']['cv_mean_auc']['mean']:.4f} ± {aggregate['summary']['cv_mean_auc']['std']:.4f}`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
