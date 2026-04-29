#!/usr/bin/env python3
"""Run native-geometry preprocessing for MU-Glioma-Post."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


EXPECTED_MODALITIES = ("t1n", "t1c", "t2f", "t2w", "tumor_mask")
CANONICAL_MODALITY_MAP = {
    "t1n": "t1",
    "t1c": "t1c",
    "t2f": "flair",
    "t2w": "t2",
}


@dataclass
class NormalizedVolume:
    modality: str
    canonical_modality: str
    image: nib.Nifti1Image
    data: np.ndarray
    source_path: Path
    output_path: Path
    clip_low: float
    clip_high: float
    foreground_mean: float
    foreground_std: float
    foreground_voxels: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create normalized full-volume and tumor-ROI outputs for MU-Glioma-Post."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("PKG-MU-Glioma-Post/MU-Glioma-Post"),
        help="Root directory containing patient folders.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("metadata/manifest.csv"),
        help="Manifest from build_phase1_audit.py.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("metadata/timepoint_summary.csv"),
        help="Timepoint summary from build_phase1_audit.py.",
    )
    parser.add_argument(
        "--splits-csv",
        type=Path,
        default=Path("metadata/splits.csv"),
        help="Patient-level split assignments.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed"),
        help="Root directory for processed outputs.",
    )
    parser.add_argument(
        "--clip-low",
        type=float,
        default=1.0,
        help="Lower percentile for nonzero-voxel clipping.",
    )
    parser.add_argument(
        "--clip-high",
        type=float,
        default=99.0,
        help="Upper percentile for nonzero-voxel clipping.",
    )
    parser.add_argument(
        "--roi-margin",
        type=int,
        default=16,
        help="Margin in voxels added around the nonzero tumor mask bbox.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional limit for testing a subset of timepoints.",
    )
    return parser.parse_args(argv)


def repo_relative(path: Path, repo_root: Path) -> str:
    absolute_path = path if path.is_absolute() else (repo_root / path)
    return absolute_path.relative_to(repo_root).as_posix()


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_nonzero(
    data: np.ndarray,
    clip_low: float,
    clip_high: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    normalized = np.zeros_like(data, dtype=np.float32)
    foreground = np.isfinite(data) & (data != 0)
    foreground_voxels = int(foreground.sum())
    if foreground_voxels == 0:
        return normalized, {
            "clip_low": float("nan"),
            "clip_high": float("nan"),
            "foreground_mean": float("nan"),
            "foreground_std": float("nan"),
            "foreground_voxels": 0,
        }

    values = data[foreground].astype(np.float32, copy=False)
    low_value = float(np.percentile(values, clip_low))
    high_value = float(np.percentile(values, clip_high))
    if high_value < low_value:
        low_value, high_value = high_value, low_value

    clipped_foreground = np.clip(values, low_value, high_value)
    mean_value = float(clipped_foreground.mean())
    std_value = float(clipped_foreground.std())
    if std_value <= 1e-8:
        normalized[foreground] = clipped_foreground - mean_value
    else:
        normalized[foreground] = (clipped_foreground - mean_value) / std_value

    return normalized, {
        "clip_low": low_value,
        "clip_high": high_value,
        "foreground_mean": mean_value,
        "foreground_std": std_value,
        "foreground_voxels": foreground_voxels,
    }


def save_nifti(
    data: np.ndarray,
    reference_image: nib.Nifti1Image,
    output_path: Path,
    affine: np.ndarray | None = None,
    dtype: np.dtype | str | type | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_affine = affine if affine is not None else reference_image.affine
    header = reference_image.header.copy()
    if dtype is not None:
        header.set_data_dtype(dtype)
    image = nib.Nifti1Image(data, final_affine, header=header)
    qform_code = int(reference_image.header.get("qform_code", 1))
    sform_code = int(reference_image.header.get("sform_code", 1))
    image.set_qform(final_affine, code=qform_code or 1)
    image.set_sform(final_affine, code=sform_code or 1)
    nib.save(image, str(output_path))


def cropped_affine(affine: np.ndarray, starts: tuple[int, int, int]) -> np.ndarray:
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = np.array(starts, dtype=np.float64)
    return affine @ translation


def bbox_from_mask(mask: np.ndarray, margin: int) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    min_coords = np.maximum(coords.min(axis=0) - margin, 0)
    max_coords = np.minimum(coords.max(axis=0) + margin + 1, np.array(mask.shape))
    starts = tuple(int(value) for value in min_coords)
    stops = tuple(int(value) for value in max_coords)
    return starts, stops


def format_triplet(values: tuple[int, int, int]) -> str:
    return "|".join(str(value) for value in values)


def process_case(
    case_rows: pd.DataFrame,
    split_name: str,
    dataset_root: Path,
    output_root: Path,
    repo_root: Path,
    clip_low: float,
    clip_high: float,
    roi_margin: int,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object], list[dict[str, object]]]:
    patient_id = str(case_rows.iloc[0]["patient_id"])
    timepoint = str(case_rows.iloc[0]["timepoint"])
    readable = case_rows[case_rows["readable"] == True].copy()
    images = readable[readable["is_mask"] == False].copy()
    masks = readable[readable["is_mask"] == True].copy()

    normalized_rows: list[dict[str, object]] = []
    normalized_volumes: dict[str, NormalizedVolume] = {}
    case_base = output_root

    for row in images.sort_values(["modality"], kind="stable").itertuples(index=False):
        source_path = dataset_root / row.file_path
        canonical_modality = CANONICAL_MODALITY_MAP.get(str(row.modality), str(row.modality))

        native_path = case_base / "images_native" / patient_id / timepoint / f"{canonical_modality}.nii.gz"
        ensure_relative_symlink(source_path, native_path)

        image = nib.load(str(source_path))
        data = np.asarray(image.dataobj, dtype=np.float32)
        normalized_data, stats = normalize_nonzero(data, clip_low=clip_low, clip_high=clip_high)

        normalized_path = case_base / "images_normalized" / patient_id / timepoint / f"{canonical_modality}.nii.gz"
        save_nifti(normalized_data.astype(np.float32), image, normalized_path, dtype=np.float32)

        normalized_volumes[canonical_modality] = NormalizedVolume(
            modality=str(row.modality),
            canonical_modality=canonical_modality,
            image=image,
            data=normalized_data.astype(np.float32, copy=False),
            source_path=source_path,
            output_path=normalized_path,
            clip_low=float(stats["clip_low"]),
            clip_high=float(stats["clip_high"]),
            foreground_mean=float(stats["foreground_mean"]),
            foreground_std=float(stats["foreground_std"]),
            foreground_voxels=int(stats["foreground_voxels"]),
        )
        normalized_rows.append(
            {
                "patient_id": patient_id,
                "timepoint": timepoint,
                "split": split_name,
                "original_modality": str(row.modality),
                "canonical_modality": canonical_modality,
                "source_file_path": repo_relative(source_path, repo_root),
                "native_link_path": repo_relative(native_path, repo_root),
                "normalized_file_path": repo_relative(normalized_path, repo_root),
                "clip_low_value": float(stats["clip_low"]),
                "clip_high_value": float(stats["clip_high"]),
                "foreground_mean": float(stats["foreground_mean"]),
                "foreground_std": float(stats["foreground_std"]),
                "foreground_voxels": int(stats["foreground_voxels"]),
            }
        )

    available_modalities = sorted(readable["modality"].tolist())
    missing_modalities = sorted(set(EXPECTED_MODALITIES) - set(available_modalities))

    mask_manifest: list[dict[str, object]] = []
    roi_status = "skipped_missing_mask"
    roi_bbox_start = ""
    roi_bbox_stop = ""
    roi_shape = ""

    if not masks.empty:
        mask_row = masks.iloc[0]
        source_mask_path = dataset_root / mask_row["file_path"]
        mask_image = nib.load(str(source_mask_path))
        mask_data = np.asarray(mask_image.dataobj)

        mask_dir = case_base / "masks" / patient_id / timepoint
        multiclass_mask_path = mask_dir / "tumor_mask_multiclass.nii.gz"
        binary_mask_path = mask_dir / "tumor_mask_binary.nii.gz"
        ensure_relative_symlink(source_mask_path, multiclass_mask_path)
        binary_mask = (mask_data > 0).astype(np.uint8)
        save_nifti(binary_mask, mask_image, binary_mask_path, dtype=np.uint8)

        bbox = bbox_from_mask(binary_mask, margin=roi_margin)
        if bbox is not None:
            starts, stops = bbox
            crop_slices = tuple(slice(start, stop) for start, stop in zip(starts, stops))
            crop_shape = tuple(int(stop - start) for start, stop in zip(starts, stops))
            crop_affine = cropped_affine(mask_image.affine, starts)
            roi_dir = case_base / "roi_tumor" / patient_id / timepoint
            radiomics_dir = case_base / "radiomics_inputs" / patient_id / timepoint

            roi_multiclass_path = roi_dir / "tumor_mask_multiclass.nii.gz"
            roi_binary_path = roi_dir / "tumor_mask_binary.nii.gz"
            cropped_mask = mask_data[crop_slices].astype(np.asarray(mask_data).dtype, copy=False)
            cropped_binary = binary_mask[crop_slices].astype(np.uint8, copy=False)
            save_nifti(cropped_mask, mask_image, roi_multiclass_path, affine=crop_affine, dtype=mask_data.dtype)
            save_nifti(cropped_binary, mask_image, roi_binary_path, affine=crop_affine, dtype=np.uint8)

            for canonical_modality, volume in normalized_volumes.items():
                roi_image_path = roi_dir / f"{canonical_modality}.nii.gz"
                cropped_image = volume.data[crop_slices].astype(np.float32, copy=False)
                save_nifti(cropped_image, volume.image, roi_image_path, affine=crop_affine, dtype=np.float32)
                radiomics_link = radiomics_dir / f"{canonical_modality}.nii.gz"
                ensure_relative_symlink(roi_image_path, radiomics_link)

            ensure_relative_symlink(roi_binary_path, radiomics_dir / "tumor_mask_binary.nii.gz")
            ensure_relative_symlink(roi_multiclass_path, radiomics_dir / "tumor_mask_multiclass.nii.gz")

            roi_status = "written"
            roi_bbox_start = format_triplet(starts)
            roi_bbox_stop = format_triplet(stops)
            roi_shape = format_triplet(crop_shape)

        label_values = sorted(int(value) for value in np.unique(mask_data))
        mask_manifest.append(
            {
                "patient_id": patient_id,
                "timepoint": timepoint,
                "split": split_name,
                "source_mask_path": repo_relative(source_mask_path, repo_root),
                "multiclass_mask_path": repo_relative(multiclass_mask_path, repo_root),
                "binary_mask_path": repo_relative(binary_mask_path, repo_root),
                "label_values": "|".join(str(value) for value in label_values),
                "tumor_voxel_count": int(binary_mask.sum()),
                "roi_status": roi_status,
                "roi_bbox_start": roi_bbox_start,
                "roi_bbox_stop": roi_bbox_stop,
                "roi_shape": roi_shape,
            }
        )

    case_row = {
        "patient_id": patient_id,
        "timepoint": timepoint,
        "split": split_name,
        "available_modalities": "|".join(available_modalities),
        "missing_modalities": "|".join(missing_modalities),
        "has_mask": bool(not masks.empty),
        "normalized_modalities_written": len(normalized_volumes),
        "roi_status": roi_status,
        "roi_bbox_start": roi_bbox_start,
        "roi_bbox_stop": roi_bbox_stop,
        "roi_shape": roi_shape,
    }
    status_row = {
        "patient_id": patient_id,
        "timepoint": timepoint,
        "split": split_name,
        "status": "processed",
        "missing_modalities": "|".join(missing_modalities),
        "notes": "mask missing" if masks.empty else "",
    }
    return normalized_rows, case_row, status_row, mask_manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path.cwd().resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()

    manifest = pd.read_csv(args.manifest_csv)
    summary = pd.read_csv(args.summary_csv)
    splits = pd.read_csv(args.splits_csv)
    split_map = {row.patient_id: row.split for row in splits.itertuples(index=False)}

    output_root.mkdir(parents=True, exist_ok=True)
    for directory_name in (
        "images_native",
        "images_reoriented",
        "images_resampled",
        "images_normalized",
        "roi_tumor",
        "masks",
        "radiomics_inputs",
        "manifests",
    ):
        (output_root / directory_name).mkdir(parents=True, exist_ok=True)

    write_text(
        output_root / "images_reoriented" / "note.txt",
        "No outputs were written because the audit found all readable volumes already aligned in LPS orientation.\n",
    )
    write_text(
        output_root / "images_resampled" / "note.txt",
        "No outputs were written because the audit found all readable volumes already sampled at 1.0 x 1.0 x 1.0 mm.\n",
    )
    write_json(
        output_root / "manifests" / "preprocessing_config.json",
        {
            "clip_low_percentile": args.clip_low,
            "clip_high_percentile": args.clip_high,
            "roi_margin_voxels": args.roi_margin,
            "canonical_modality_map": CANONICAL_MODALITY_MAP,
            "dataset_root": repo_relative(dataset_root, repo_root),
            "phase_3_decision": "keep native geometry; normalize intensities; preserve multi-class masks; derive ROI crops when masks exist",
        },
    )

    case_keys = summary[["patient_id", "timepoint"]].drop_duplicates().sort_values(
        ["patient_id", "timepoint"],
        kind="stable",
    )
    if args.max_cases is not None:
        case_keys = case_keys.head(args.max_cases)

    normalized_manifest_rows: list[dict[str, object]] = []
    case_manifest_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    mask_manifest_rows: list[dict[str, object]] = []

    total_cases = len(case_keys)
    for index, key in enumerate(case_keys.itertuples(index=False), start=1):
        case_rows = manifest[
            (manifest["patient_id"] == key.patient_id)
            & (manifest["timepoint"] == key.timepoint)
        ].copy()
        split_name = split_map.get(key.patient_id, "")
        norm_rows, case_row, status_row, mask_rows = process_case(
            case_rows=case_rows,
            split_name=split_name,
            dataset_root=dataset_root,
            output_root=output_root,
            repo_root=repo_root,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            roi_margin=args.roi_margin,
        )
        normalized_manifest_rows.extend(norm_rows)
        case_manifest_rows.append(case_row)
        status_rows.append(status_row)
        mask_manifest_rows.extend(mask_rows)

        if index % 25 == 0 or index == total_cases:
            print(f"Processed {index}/{total_cases} timepoints", flush=True)

    normalized_manifest = pd.DataFrame(normalized_manifest_rows).sort_values(
        ["patient_id", "timepoint", "canonical_modality"],
        kind="stable",
    )
    case_manifest = pd.DataFrame(case_manifest_rows).sort_values(
        ["patient_id", "timepoint"],
        kind="stable",
    )
    status_manifest = pd.DataFrame(status_rows).sort_values(
        ["patient_id", "timepoint"],
        kind="stable",
    )
    mask_manifest = pd.DataFrame(mask_manifest_rows).sort_values(
        ["patient_id", "timepoint"],
        kind="stable",
    )

    normalized_manifest.to_csv(output_root / "manifests" / "normalized_manifest.csv", index=False)
    case_manifest.to_csv(output_root / "manifests" / "case_manifest.csv", index=False)
    status_manifest.to_csv(output_root / "manifests" / "preprocessing_status.csv", index=False)
    mask_manifest.to_csv(output_root / "manifests" / "mask_manifest.csv", index=False)

    print(f"Wrote {output_root / 'manifests' / 'normalized_manifest.csv'}")
    print(f"Wrote {output_root / 'manifests' / 'case_manifest.csv'}")
    print(f"Wrote {output_root / 'manifests' / 'preprocessing_status.csv'}")
    print(f"Wrote {output_root / 'manifests' / 'mask_manifest.csv'}")


if __name__ == "__main__":
    main()
