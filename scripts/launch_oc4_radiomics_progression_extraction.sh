#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${1:-oc4}"
REMOTE_REPO="/project/community/sbandred/mu-glioma"
REMOTE_LOG="/tmp/mu_glioma_radiomics_progression_extract.log"
REMOTE_PID="/tmp/mu_glioma_radiomics_progression_extract.pid"

ssh "${REMOTE_HOST}" "bash -lc '
cd ${REMOTE_REPO}
mkdir -p results/radiomics_baseline_progression
nohup .venv/bin/python scripts/run_radiomics_baseline.py \
  --extract-only \
  --workers 24 \
  --output-dir results/radiomics_baseline_progression \
  --target-column clinical_progression \
  > ${REMOTE_LOG} 2>&1 < /dev/null &
echo \$! > ${REMOTE_PID}
cat ${REMOTE_PID}
'"

echo "Remote log: ${REMOTE_LOG}"
echo "Remote pid file: ${REMOTE_PID}"
