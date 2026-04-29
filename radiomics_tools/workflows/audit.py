#!/usr/bin/env python3
"""Build Phase 1 audit artifacts for the MU-Glioma-Post dataset."""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd


EXPECTED_MODALITIES = ("t1n", "t1c", "t2f", "t2w", "tumor_mask")
REFERENCE_MODALITY_ORDER = ("t1c", "t1n", "t2f", "t2w")


@dataclass(frozen=True)
class ParsedPath:
    patient_id: str
    timepoint: str
    file_name: str
    file_path: str
    series_label: str
    modality: str
    is_mask: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manifest and consistency reports for MU-Glioma-Post."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("PKG-MU-Glioma-Post/MU-Glioma-Post"),
        help="Root directory containing PatientID_xxxx folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata"),
        help="Directory where CSV artifacts will be written.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Reuse an existing manifest CSV and rebuild only the summary reports.",
    )
    return parser.parse_args()


def format_sequence(values: Iterable[object], precision: int = 6) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, (float, np.floating)):
            if math.isnan(float(value)):
                parts.append("nan")
            else:
                parts.append(f"{float(value):.{precision}f}")
        else:
            parts.append(str(value))
    return "|".join(parts)


def canonical_modality(series_label: str) -> tuple[str, bool]:
    if series_label == "tumorMask":
        return "tumor_mask", True
    if series_label.startswith("brain_"):
        return series_label.removeprefix("brain_"), False
    return series_label.lower(), False


def parse_nifti_path(dataset_root: Path, file_path: Path) -> ParsedPath:
    relative_path = file_path.relative_to(dataset_root)
    patient_id, timepoint, file_name = relative_path.parts
    series_label = file_name.removesuffix(".nii.gz").split(f"{timepoint}_", 1)[-1]
    modality, is_mask = canonical_modality(series_label)
    return ParsedPath(
        patient_id=patient_id,
        timepoint=timepoint,
        file_name=file_name,
        file_path=relative_path.as_posix(),
        series_label=series_label,
        modality=modality,
        is_mask=is_mask,
    )


def affine_hash(affine: np.ndarray) -> str:
    rounded = np.round(np.asarray(affine, dtype=np.float64), 6)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def affine_summary(affine: np.ndarray) -> str:
    rounded = np.round(np.asarray(affine, dtype=np.float64), 6)
    return ";".join(format_sequence(row, precision=6) for row in rounded)


def safe_unique_mask_values(data: np.ndarray) -> str:
    values = np.unique(data)
    if values.size > 32:
        return f"{values.size}_unique_values"
    return format_sequence(values.tolist(), precision=6)


def mask_values_sane(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    if not np.all(np.isfinite(values)):
        return False
    rounded = np.round(values)
    return bool(np.allclose(values, rounded) and np.min(values) >= 0)


def coerce_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda value: bool(value) if pd.notna(value) else False)


def audit_file(dataset_root: Path, file_path: Path) -> dict[str, object]:
    parsed = parse_nifti_path(dataset_root, file_path)
    base_record: dict[str, object] = {
        "patient_id": parsed.patient_id,
        "timepoint": parsed.timepoint,
        "file_name": parsed.file_name,
        "file_path": parsed.file_path,
        "series_label": parsed.series_label,
        "modality": parsed.modality,
        "is_mask": parsed.is_mask,
        "readable": False,
        "shape": "",
        "spacing": "",
        "dtype": "",
        "orientation": "",
        "affine_hash": "",
        "affine_summary": "",
        "min_intensity": np.nan,
        "max_intensity": np.nan,
        "mean_intensity": np.nan,
        "std_intensity": np.nan,
        "nonzero_fraction": np.nan,
        "nonzero_voxel_count": np.nan,
        "mask_unique_values": "",
        "mask_value_count": np.nan,
        "mask_is_binary": np.nan,
        "mask_values_sane": np.nan,
        "is_empty_mask": np.nan,
        "error": "",
    }

    try:
        image = nib.load(str(file_path))
        data = np.asanyarray(image.dataobj)
    except Exception as exc:  # pragma: no cover - depends on dataset corruption.
        base_record["error"] = f"{type(exc).__name__}: {exc}"
        return base_record

    finite_mask = np.isfinite(data)
    finite_values = data[finite_mask]
    voxel_count = int(data.size)
    nonzero_voxel_count = int(np.count_nonzero(data))

    shape = tuple(int(dim) for dim in image.shape)
    zooms = tuple(float(value) for value in image.header.get_zooms()[: len(shape)])
    orientation = "".join(nib.aff2axcodes(image.affine))
    unique_values = np.unique(finite_values) if parsed.is_mask else np.array([], dtype=data.dtype)

    min_intensity = float(finite_values.min()) if finite_values.size else np.nan
    max_intensity = float(finite_values.max()) if finite_values.size else np.nan
    mean_intensity = float(finite_values.mean()) if finite_values.size else np.nan
    std_intensity = float(finite_values.std()) if finite_values.size else np.nan

    mask_is_binary = np.nan
    mask_sane = np.nan
    is_empty_mask = np.nan
    unique_value_summary = ""
    unique_value_count = np.nan
    if parsed.is_mask:
        mask_is_binary = bool(set(unique_values.tolist()).issubset({0, 1}))
        mask_sane = mask_values_sane(unique_values)
        is_empty_mask = bool(nonzero_voxel_count == 0)
        unique_value_summary = safe_unique_mask_values(unique_values)
        unique_value_count = int(unique_values.size)

    return {
        **base_record,
        "readable": True,
        "shape": format_sequence(shape, precision=0),
        "spacing": format_sequence(zooms, precision=6),
        "dtype": np.dtype(image.get_data_dtype()).name,
        "orientation": orientation,
        "affine_hash": affine_hash(image.affine),
        "affine_summary": affine_summary(image.affine),
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "nonzero_fraction": (nonzero_voxel_count / voxel_count) if voxel_count else np.nan,
        "nonzero_voxel_count": nonzero_voxel_count,
        "mask_unique_values": unique_value_summary,
        "mask_value_count": unique_value_count,
        "mask_is_binary": mask_is_binary,
        "mask_values_sane": mask_sane,
        "is_empty_mask": is_empty_mask,
    }


def choose_reference_image(image_rows: pd.DataFrame) -> pd.Series | None:
    for modality in REFERENCE_MODALITY_ORDER:
        matches = image_rows[image_rows["modality"] == modality]
        if not matches.empty:
            return matches.iloc[0]
    if image_rows.empty:
        return None
    return image_rows.iloc[0]


def summarize_timepoint(group: pd.DataFrame) -> tuple[dict[str, object], list[dict[str, object]]]:
    patient_id = str(group.iloc[0]["patient_id"])
    timepoint = str(group.iloc[0]["timepoint"])
    readable = group[group["readable"] == True]
    images = readable[readable["is_mask"] == False]
    masks = readable[readable["is_mask"] == True]

    present_modalities = sorted(set(readable["modality"].tolist()))
    missing_modalities = sorted(set(EXPECTED_MODALITIES) - set(present_modalities))
    unexpected_modalities = sorted(set(present_modalities) - set(EXPECTED_MODALITIES))

    image_shape_consistent = bool(images["shape"].nunique(dropna=False) <= 1) if not images.empty else False
    image_spacing_consistent = bool(images["spacing"].nunique(dropna=False) <= 1) if not images.empty else False
    image_orientation_consistent = bool(images["orientation"].nunique(dropna=False) <= 1) if not images.empty else False
    image_affine_consistent = bool(images["affine_hash"].nunique(dropna=False) <= 1) if not images.empty else False

    reference = choose_reference_image(images)
    mask_shape_aligned = np.nan
    mask_spacing_aligned = np.nan
    mask_orientation_aligned = np.nan
    mask_affine_aligned = np.nan
    if reference is not None and not masks.empty:
        mask_shape_aligned = bool((masks["shape"] == reference["shape"]).all())
        mask_spacing_aligned = bool((masks["spacing"] == reference["spacing"]).all())
        mask_orientation_aligned = bool((masks["orientation"] == reference["orientation"]).all())
        mask_affine_aligned = bool((masks["affine_hash"] == reference["affine_hash"]).all())

    issues: list[dict[str, object]] = []

    def add_issue(issue_code: str, issue_detail: str) -> None:
        issues.append(
            {
                "patient_id": patient_id,
                "timepoint": timepoint,
                "issue_code": issue_code,
                "issue_detail": issue_detail,
                "present_modalities": format_sequence(present_modalities),
                "missing_modalities": format_sequence(missing_modalities),
                "unexpected_modalities": format_sequence(unexpected_modalities),
                "unreadable_files": int((group["readable"] == False).sum()),
            }
        )

    if missing_modalities:
        add_issue("missing_modality", f"Missing expected modalities: {format_sequence(missing_modalities)}")
    if unexpected_modalities:
        add_issue(
            "unexpected_modality",
            f"Unexpected modalities present: {format_sequence(unexpected_modalities)}",
        )
    if (group["readable"] == False).any():
        bad_files = group.loc[group["readable"] == False, "file_name"].tolist()
        add_issue("unreadable_file", f"Unreadable files: {format_sequence(sorted(bad_files))}")
    if len(masks) == 0:
        add_issue("missing_mask", "No readable tumor mask found")
    if len(masks) > 1:
        add_issue("multiple_masks", f"Found {len(masks)} readable tumor masks")
    if not images.empty and not image_shape_consistent:
        add_issue("image_shape_mismatch", "Image modalities have inconsistent shapes")
    if not images.empty and not image_spacing_consistent:
        add_issue("image_spacing_mismatch", "Image modalities have inconsistent voxel spacing")
    if not images.empty and not image_orientation_consistent:
        add_issue("image_orientation_mismatch", "Image modalities have inconsistent orientation codes")
    if not images.empty and not image_affine_consistent:
        add_issue("image_affine_mismatch", "Image modalities have inconsistent affine transforms")
    if reference is not None and not masks.empty and not bool(mask_shape_aligned):
        add_issue("mask_shape_mismatch", "Tumor mask shape does not match the reference image")
    if reference is not None and not masks.empty and not bool(mask_spacing_aligned):
        add_issue("mask_spacing_mismatch", "Tumor mask spacing does not match the reference image")
    if reference is not None and not masks.empty and not bool(mask_orientation_aligned):
        add_issue("mask_orientation_mismatch", "Tumor mask orientation does not match the reference image")
    if reference is not None and not masks.empty and not bool(mask_affine_aligned):
        add_issue("mask_affine_mismatch", "Tumor mask affine does not match the reference image")
    empty_mask_flags = coerce_bool(masks["is_empty_mask"]) if not masks.empty else pd.Series(dtype=bool)
    sane_mask_flags = coerce_bool(masks["mask_values_sane"]) if not masks.empty else pd.Series(dtype=bool)
    binary_mask_flags = coerce_bool(masks["mask_is_binary"]) if not masks.empty else pd.Series(dtype=bool)

    if not masks.empty and empty_mask_flags.any():
        empty_mask_files = masks.loc[empty_mask_flags, "file_name"].tolist()
        add_issue("empty_mask", f"Empty masks: {format_sequence(sorted(empty_mask_files))}")
    if not masks.empty and (~sane_mask_flags).any():
        bad_mask_files = masks.loc[~sane_mask_flags, "file_name"].tolist()
        add_issue("mask_values_insane", f"Non-integer or negative mask values: {format_sequence(sorted(bad_mask_files))}")

    summary = {
        "patient_id": patient_id,
        "timepoint": timepoint,
        "num_files": int(len(group)),
        "num_readable_files": int(len(readable)),
        "present_modalities": format_sequence(present_modalities),
        "missing_modalities": format_sequence(missing_modalities),
        "unexpected_modalities": format_sequence(unexpected_modalities),
        "has_mask": bool(len(masks) > 0),
        "mask_file_count": int(len(masks)),
        "image_file_count": int(len(images)),
        "image_shape_consistent": image_shape_consistent,
        "image_spacing_consistent": image_spacing_consistent,
        "image_orientation_consistent": image_orientation_consistent,
        "image_affine_consistent": image_affine_consistent,
        "mask_shape_aligned": mask_shape_aligned,
        "mask_spacing_aligned": mask_spacing_aligned,
        "mask_orientation_aligned": mask_orientation_aligned,
        "mask_affine_aligned": mask_affine_aligned,
        "has_empty_mask": bool(empty_mask_flags.any()) if not masks.empty else False,
        "has_multiclass_mask": bool((~binary_mask_flags).any()) if not masks.empty else False,
        "has_insane_mask_values": bool((~sane_mask_flags).any()) if not masks.empty else False,
        "issue_count": len(issues),
        "issue_codes": format_sequence(issue["issue_code"] for issue in issues),
    }
    return summary, issues


def sort_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest.sort_values(
        ["patient_id", "timepoint", "is_mask", "modality", "file_name"],
        kind="stable",
    )


def build_manifest(dataset_root: Path) -> pd.DataFrame:
    nifti_paths = sorted(dataset_root.rglob("*.nii.gz"))
    if not nifti_paths:
        raise SystemExit(f"No NIfTI files found under {dataset_root}")

    manifest_rows: list[dict[str, object]] = []
    for index, file_path in enumerate(nifti_paths, start=1):
        manifest_rows.append(audit_file(dataset_root, file_path))
        if index % 250 == 0 or index == len(nifti_paths):
            print(f"Audited {index}/{len(nifti_paths)} files", flush=True)

    return sort_manifest(pd.DataFrame(manifest_rows))


def build_summary_tables(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = sort_manifest(manifest)

    summary_rows: list[dict[str, object]] = []
    missingness_rows: list[dict[str, object]] = []
    for _, group in manifest.groupby(["patient_id", "timepoint"], sort=True):
        summary_row, issues = summarize_timepoint(group.reset_index(drop=True))
        summary_rows.append(summary_row)
        missingness_rows.extend(issues)

    timepoint_summary = pd.DataFrame(summary_rows).sort_values(["patient_id", "timepoint"], kind="stable")
    missingness_report = (
        pd.DataFrame(missingness_rows).sort_values(["patient_id", "timepoint", "issue_code"], kind="stable")
        if missingness_rows
        else pd.DataFrame(
            columns=[
                "patient_id",
                "timepoint",
                "issue_code",
                "issue_detail",
                "present_modalities",
                "missing_modalities",
                "unexpected_modalities",
                "unreadable_files",
            ]
        )
    )
    return timepoint_summary, missingness_report


def build_outputs(dataset_root: Path, output_dir: Path, manifest_csv: Path | None = None) -> None:
    if manifest_csv is None:
        manifest = build_manifest(dataset_root)
    else:
        manifest = pd.read_csv(manifest_csv)

    timepoint_summary, missingness_report = build_summary_tables(manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    timepoint_summary.to_csv(output_dir / "timepoint_summary.csv", index=False)
    missingness_report.to_csv(output_dir / "missingness_report.csv", index=False)

    print(f"Wrote {output_dir / 'manifest.csv'}")
    print(f"Wrote {output_dir / 'timepoint_summary.csv'}")
    print(f"Wrote {output_dir / 'missingness_report.csv'}")


def main() -> None:
    args = parse_args()
    manifest_csv = args.manifest_csv.resolve() if args.manifest_csv else None
    build_outputs(args.dataset_root.resolve(), args.output_dir.resolve(), manifest_csv=manifest_csv)


if __name__ == "__main__":
    main()
