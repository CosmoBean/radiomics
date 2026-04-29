#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT_INDEX="${EXPERIMENT_INDEX:-processed/manifests/experiment_index.csv}"
RADIOMICS_YAML="${RADIOMICS_YAML:-configs/postoperative_progression_surveillance_radiomics.yaml}"
RESULT_DIR="${RESULT_DIR:-results/calibrated_forward_hybrid}"
MODEL_DIR="${MODEL_DIR:-models/calibrated}"
SEED="${SEED:-62}"

"$PYTHON_BIN" -m radiomics_tools.workflows.train \
  --experiment-index "$EXPERIMENT_INDEX" \
  --radiomics-yaml "$RADIOMICS_YAML" \
  --output-dir "$RESULT_DIR" \
  --label-mode within_window \
  --progression-window-days 120 \
  --pre-progression-only \
  --exclude-after-late-treatment \
  --models lightgbm,logreg,rf,svm \
  --modalities t1c,flair \
  --clinical-feature-set hybrid_basic \
  --feature-subsets 16,24,32,48 \
  --n-trials 25 \
  --ranking-folds 5 \
  --cv-folds 5 \
  --permutation-repeats 20 \
  --bootstrap-iterations 300 \
  --progress-bar off \
  --seed "$SEED"

"$PYTHON_BIN" -m radiomics_tools.workflows.export_calibrated \
  --result-dir "$RESULT_DIR" \
  --output-dir "$MODEL_DIR" \
  --seed "$SEED" \
  --cv-folds 5
