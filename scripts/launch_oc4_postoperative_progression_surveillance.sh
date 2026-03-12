#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${1:-oc4}"
REMOTE_REPO="/project/community/$(whoami)/mu-glioma"
REMOTE_LOG="/tmp/mu_glioma_postoperative_progression_surveillance.log"
REMOTE_PID="/tmp/mu_glioma_postoperative_progression_surveillance.pid"

ssh "${REMOTE_HOST}" "bash -lc '
cd ${REMOTE_REPO}
mkdir -p results/postoperative_progression_surveillance
nohup .venv/bin/python scripts/run_postoperative_progression_surveillance.py \
  --output-dir results/postoperative_progression_surveillance \
  --cache-root processed/postoperative_progression_surveillance \
  --label-mode post_progression \
  --models lightgbm \
  --n-trials 80 \
  --max-workers 24 \
  --reuse-case-features \
  > ${REMOTE_LOG} 2>&1 < /dev/null &
echo \$! > ${REMOTE_PID}
cat ${REMOTE_PID}
'"

echo "Remote log: ${REMOTE_LOG}"
echo "Remote pid file: ${REMOTE_PID}"
