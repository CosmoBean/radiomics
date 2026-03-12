#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${1:-oc4}"
REMOTE_REPO="/project/community/sbandred/mu-glioma"
REMOTE_LOG="/tmp/mu_glioma_explainable_grade4_auto.log"
REMOTE_PID="/tmp/mu_glioma_explainable_grade4_auto.pid"

ssh "${REMOTE_HOST}" "bash -lc '
cd ${REMOTE_REPO}
mkdir -p results/explainable_radiomics_grade4_auto
nohup .venv/bin/python scripts/run_explainable_radiomics_search.py \
  --max-iters 25 \
  --max-runtime-seconds 3600 \
  --output-dir results/explainable_radiomics_grade4_auto \
  --target grade4_vs_lower \
  > ${REMOTE_LOG} 2>&1 < /dev/null &
echo \$! > ${REMOTE_PID}
cat ${REMOTE_PID}
'"

echo "Remote log: ${REMOTE_LOG}"
echo "Remote pid file: ${REMOTE_PID}"
