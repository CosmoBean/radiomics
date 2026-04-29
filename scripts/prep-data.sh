#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_ROOT="${DATASET_ROOT:-PKG-MU-Glioma-Post/MU-Glioma-Post}"
CLINICAL_XLSX="${CLINICAL_XLSX:-PKG-MU-Glioma-Post/MU-Glioma-Post_ClinicalData-July2025.xlsx}"
PROCESSED_ROOT="${PROCESSED_ROOT:-processed}"

"$PYTHON_BIN" -m radiomics_tools.workflows.audit \
  --dataset-root "$DATASET_ROOT" \
  --output-dir metadata

"$PYTHON_BIN" -m radiomics_tools.workflows.split_patients \
  --summary-csv metadata/timepoint_summary.csv \
  --output-dir metadata

"$PYTHON_BIN" -m radiomics_tools.workflows.preprocess \
  --dataset-root "$DATASET_ROOT" \
  --manifest-csv metadata/manifest.csv \
  --summary-csv metadata/timepoint_summary.csv \
  --splits-csv metadata/splits.csv \
  --output-root "$PROCESSED_ROOT"

"$PYTHON_BIN" -m radiomics_tools.workflows.build_index \
  --clinical-xlsx "$CLINICAL_XLSX" \
  --processed-root "$PROCESSED_ROOT" \
  --output-dir "$PROCESSED_ROOT/manifests"
