#!/usr/bin/env python3
"""Build experiment-ready cohort indices from processed outputs and spreadsheets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


ROI_MODALITIES = ("t1", "t1c", "flair", "t2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build merged clinical/imaging indices for MU-Glioma-Post."
    )
    parser.add_argument(
        "--clinical-xlsx",
        type=Path,
        default=Path("PKG-MU-Glioma-Post/MU-Glioma-Post_ClinicalData-July2025.xlsx"),
        help="Clinical workbook.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("processed"),
        help="Processed output root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/manifests"),
        help="Directory where merged indices will be written.",
    )
    return parser.parse_args()


def snake_case(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def repo_relative(path: Path, repo_root: Path) -> str:
    absolute_path = path if path.is_absolute() else (repo_root / path)
    return absolute_path.relative_to(repo_root).as_posix()


def load_clinical_table(clinical_xlsx: Path) -> tuple[pd.DataFrame, dict[int, str]]:
    clinical = pd.read_excel(clinical_xlsx, sheet_name="MU Glioma Post")
    rename_map = {}
    for column in clinical.columns:
        if str(column) == "Patient_ID":
            rename_map[column] = "patient_id"
        else:
            rename_map[column] = f"clinical_{snake_case(str(column))}"
    clinical = clinical.rename(columns=rename_map)

    timepoint_day_columns: dict[int, str] = {}
    for original_name, renamed_name in rename_map.items():
        match = re.search(r"Timepoint_(\d+)", str(original_name))
        if match:
            timepoint_day_columns[int(match.group(1))] = renamed_name

    return clinical, timepoint_day_columns


def compute_label_voxel_counts(mask_manifest: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in mask_manifest.itertuples(index=False):
        mask_path = repo_root / row.source_mask_path
        data = np.asarray(nib.load(str(mask_path)).dataobj)
        unique_values, counts = np.unique(data, return_counts=True)
        count_map = {int(value): int(count) for value, count in zip(unique_values.tolist(), counts.tolist())}
        rows.append(
            {
                "patient_id": row.patient_id,
                "timepoint": row.timepoint,
                "label_values_present": "|".join(str(int(value)) for value in unique_values.tolist()),
                "label1_voxels": count_map.get(1, 0),
                "label2_voxels": count_map.get(2, 0),
                "label3_voxels": count_map.get(3, 0),
                "label4_voxels": count_map.get(4, 0),
                "whole_tumor_voxels": int(sum(count for value, count in count_map.items() if value > 0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    processed_root = args.processed_root.resolve()
    output_dir = args.output_dir.resolve()

    case_manifest = pd.read_csv(processed_root / "manifests" / "case_manifest.csv")
    normalized_manifest = pd.read_csv(processed_root / "manifests" / "normalized_manifest.csv")
    mask_manifest = pd.read_csv(processed_root / "manifests" / "mask_manifest.csv")
    clinical, timepoint_day_columns = load_clinical_table(args.clinical_xlsx.resolve())

    label_counts = compute_label_voxel_counts(mask_manifest, repo_root)

    normalized_pivot = (
        normalized_manifest.pivot_table(
            index=["patient_id", "timepoint"],
            columns="canonical_modality",
            values="normalized_file_path",
            aggfunc="first",
        )
        .rename(columns=lambda value: f"normalized_{value}_path")
        .reset_index()
    )
    native_pivot = (
        normalized_manifest.pivot_table(
            index=["patient_id", "timepoint"],
            columns="canonical_modality",
            values="native_link_path",
            aggfunc="first",
        )
        .rename(columns=lambda value: f"native_{value}_path")
        .reset_index()
    )

    mask_paths = mask_manifest[
        [
            "patient_id",
            "timepoint",
            "split",
            "source_mask_path",
            "multiclass_mask_path",
            "binary_mask_path",
            "label_values",
            "tumor_voxel_count",
        ]
    ].rename(
        columns={
            "label_values": "mask_label_values",
            "tumor_voxel_count": "mask_tumor_voxel_count",
        }
    )

    case_index = (
        case_manifest.merge(normalized_pivot, on=["patient_id", "timepoint"], how="left")
        .merge(native_pivot, on=["patient_id", "timepoint"], how="left")
        .merge(mask_paths, on=["patient_id", "timepoint", "split"], how="left")
        .merge(label_counts, on=["patient_id", "timepoint"], how="left")
        .merge(clinical, on="patient_id", how="left")
    )

    case_index["timepoint_number"] = case_index["timepoint"].str.extract(r"Timepoint_(\d+)").astype(int)
    case_index["days_from_diagnosis_to_mri"] = case_index.apply(
        lambda row: row.get(timepoint_day_columns.get(int(row["timepoint_number"]))) if int(row["timepoint_number"]) in timepoint_day_columns else np.nan,
        axis=1,
    )

    for modality in ROI_MODALITIES:
        case_index[f"roi_{modality}_path"] = case_index.apply(
            lambda row, modality=modality: (
                repo_relative(processed_root / "roi_tumor" / row["patient_id"] / row["timepoint"] / f"{modality}.nii.gz", repo_root)
                if row["roi_status"] == "written"
                else ""
            ),
            axis=1,
        )
        case_index[f"radiomics_{modality}_path"] = case_index.apply(
            lambda row, modality=modality: (
                repo_relative(processed_root / "radiomics_inputs" / row["patient_id"] / row["timepoint"] / f"{modality}.nii.gz", repo_root)
                if row["roi_status"] == "written"
                else ""
            ),
            axis=1,
        )

    case_index["roi_binary_mask_path"] = case_index.apply(
        lambda row: (
            repo_relative(processed_root / "roi_tumor" / row["patient_id"] / row["timepoint"] / "tumor_mask_binary.nii.gz", repo_root)
            if row["roi_status"] == "written"
            else ""
        ),
        axis=1,
    )
    case_index["roi_multiclass_mask_path"] = case_index.apply(
        lambda row: (
            repo_relative(processed_root / "roi_tumor" / row["patient_id"] / row["timepoint"] / "tumor_mask_multiclass.nii.gz", repo_root)
            if row["roi_status"] == "written"
            else ""
        ),
        axis=1,
    )
    case_index["radiomics_binary_mask_path"] = case_index.apply(
        lambda row: (
            repo_relative(processed_root / "radiomics_inputs" / row["patient_id"] / row["timepoint"] / "tumor_mask_binary.nii.gz", repo_root)
            if row["roi_status"] == "written"
            else ""
        ),
        axis=1,
    )
    case_index["radiomics_multiclass_mask_path"] = case_index.apply(
        lambda row: (
            repo_relative(processed_root / "radiomics_inputs" / row["patient_id"] / row["timepoint"] / "tumor_mask_multiclass.nii.gz", repo_root)
            if row["roi_status"] == "written"
            else ""
        ),
        axis=1,
    )

    longitudinal_index = case_index[
        [
            "patient_id",
            "timepoint",
            "timepoint_number",
            "split",
            "days_from_diagnosis_to_mri",
            "has_mask",
            "whole_tumor_voxels",
            "label1_voxels",
            "label2_voxels",
            "label3_voxels",
            "label4_voxels",
        ]
    ].sort_values(["patient_id", "timepoint_number"], kind="stable")

    timepoint_day_columns_present = [column for _, column in sorted(timepoint_day_columns.items())]
    clinical_alignment = clinical[["patient_id", *timepoint_day_columns_present]].copy()
    clinical_alignment["clinical_timepoint_count"] = clinical_alignment[timepoint_day_columns_present].notna().sum(axis=1)
    imaging_counts = case_manifest.groupby("patient_id").size().rename("imaging_timepoint_count").reset_index()
    clinical_alignment = clinical_alignment.merge(imaging_counts, on="patient_id", how="left")
    clinical_alignment["imaging_timepoint_count"] = clinical_alignment["imaging_timepoint_count"].fillna(0).astype(int)
    clinical_alignment["timepoint_count_match"] = (
        clinical_alignment["clinical_timepoint_count"] == clinical_alignment["imaging_timepoint_count"]
    )

    patient_clinical_table = clinical.sort_values("patient_id", kind="stable")
    case_index = case_index.sort_values(["patient_id", "timepoint_number"], kind="stable")

    output_dir.mkdir(parents=True, exist_ok=True)
    patient_clinical_table.to_csv(repo_root / "metadata" / "patient_clinical_table.csv", index=False)
    case_index.to_csv(output_dir / "experiment_index.csv", index=False)
    longitudinal_index.to_csv(output_dir / "longitudinal_index.csv", index=False)
    clinical_alignment.to_csv(output_dir / "clinical_alignment_report.csv", index=False)

    print(f"Wrote {repo_relative(repo_root / 'metadata' / 'patient_clinical_table.csv', repo_root)}")
    print(f"Wrote {repo_relative(output_dir / 'experiment_index.csv', repo_root)}")
    print(f"Wrote {repo_relative(output_dir / 'longitudinal_index.csv', repo_root)}")
    print(f"Wrote {repo_relative(output_dir / 'clinical_alignment_report.csv', repo_root)}")


if __name__ == "__main__":
    main()
