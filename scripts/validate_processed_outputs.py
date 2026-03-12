#!/usr/bin/env python3
"""Validate processed MU-Glioma-Post outputs against source metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate processed outputs and write a summary JSON report."
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("metadata/manifest.csv"),
        help="Source manifest from build_phase1_audit.py.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("processed"),
        help="Processed output root.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=12,
        help="Number of normalized files to sample for header/data checks.",
    )
    return parser.parse_args()


def file_exists_series(paths: pd.Series, root: Path) -> pd.Series:
    return paths.map(lambda path: (root / path).exists())


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    processed_root = args.processed_root.resolve()

    source_manifest = pd.read_csv(args.source_manifest)
    normalized_manifest = pd.read_csv(processed_root / "manifests" / "normalized_manifest.csv")
    case_manifest = pd.read_csv(processed_root / "manifests" / "case_manifest.csv")
    mask_manifest = pd.read_csv(processed_root / "manifests" / "mask_manifest.csv")

    source_images = source_manifest[(source_manifest["readable"] == True) & (source_manifest["is_mask"] == False)]
    source_masks = source_manifest[(source_manifest["readable"] == True) & (source_manifest["is_mask"] == True)]

    native_exists = normalized_manifest["native_link_path"].map(lambda path: (repo_root / path).is_symlink())
    normalized_exists = normalized_manifest["normalized_file_path"].map(lambda path: (repo_root / path).exists())
    multiclass_exists = mask_manifest["multiclass_mask_path"].map(lambda path: (repo_root / path).exists())
    binary_exists = mask_manifest["binary_mask_path"].map(lambda path: (repo_root / path).exists())

    sample_count = min(args.sample_count, len(normalized_manifest))
    sampled = normalized_manifest.sample(n=sample_count, random_state=20260310) if sample_count else normalized_manifest

    sampled_checks: list[dict[str, object]] = []
    for row in sampled.itertuples(index=False):
        image = nib.load(str(repo_root / row.normalized_file_path))
        data = np.asarray(image.dataobj, dtype=np.float32)
        nonzero = data[data != 0]
        sampled_checks.append(
            {
                "normalized_file_path": row.normalized_file_path,
                "shape": list(map(int, data.shape)),
                "dtype": str(data.dtype),
                "nonzero_voxels": int(nonzero.size),
                "finite": bool(np.isfinite(data).all()),
                "nonzero_mean": float(nonzero.mean()) if nonzero.size else 0.0,
                "nonzero_std": float(nonzero.std()) if nonzero.size else 0.0,
            }
        )

    summary = {
        "processed_root": processed_root.relative_to(repo_root).as_posix(),
        "source_readable_images": int(len(source_images)),
        "source_readable_masks": int(len(source_masks)),
        "processed_normalized_images": int(len(normalized_manifest)),
        "processed_cases": int(len(case_manifest)),
        "processed_masks": int(len(mask_manifest)),
        "missing_mask_cases": int((case_manifest["has_mask"] == False).sum()),
        "roi_status_counts": case_manifest["roi_status"].value_counts(dropna=False).to_dict(),
        "native_symlinks_present": int(native_exists.sum()),
        "normalized_files_present": int(normalized_exists.sum()),
        "multiclass_masks_present": int(multiclass_exists.sum()),
        "binary_masks_present": int(binary_exists.sum()),
        "sampled_normalized_checks": sampled_checks,
    }

    output_path = processed_root / "manifests" / "validation_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
