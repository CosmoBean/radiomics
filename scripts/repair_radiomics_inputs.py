#!/usr/bin/env python3
"""Rebuild processed/radiomics_inputs from processed/roi_tumor."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


EXPECTED_FILES = (
    "t1.nii.gz",
    "t1c.nii.gz",
    "flair.nii.gz",
    "t2.nii.gz",
    "tumor_mask_binary.nii.gz",
    "tumor_mask_multiclass.nii.gz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate radiomics_inputs as symlinks to roi_tumor files."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("processed"),
        help="Processed dataset root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing radiomics_inputs tree before rebuilding it.",
    )
    return parser.parse_args()


def reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_relative_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        reset_path(link_path)
    relative_target = os.path.relpath(target.resolve(), link_path.parent.resolve())
    link_path.symlink_to(relative_target)


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root.resolve()
    roi_root = processed_root / "roi_tumor"
    radiomics_root = processed_root / "radiomics_inputs"

    if not roi_root.exists():
        raise SystemExit(f"ROI tree not found: {roi_root}")

    if radiomics_root.exists():
        if not args.force:
            raise SystemExit(
                f"{radiomics_root} already exists; rerun with --force to rebuild it."
            )
        reset_path(radiomics_root)

    linked_files = 0
    case_count = 0
    missing_files: list[str] = []

    for case_dir in sorted(path for path in roi_root.glob("PatientID_*/*") if path.is_dir()):
        case_count += 1
        relative_case_dir = case_dir.relative_to(roi_root)
        target_case_dir = radiomics_root / relative_case_dir
        for file_name in EXPECTED_FILES:
            source_path = case_dir / file_name
            if not source_path.exists():
                missing_files.append(source_path.relative_to(processed_root).as_posix())
                continue
            ensure_relative_symlink(source_path, target_case_dir / file_name)
            linked_files += 1

    print(f"Processed {case_count} ROI cases")
    print(f"Created {linked_files} symlinks under {radiomics_root}")
    if missing_files:
        print("Missing source files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
