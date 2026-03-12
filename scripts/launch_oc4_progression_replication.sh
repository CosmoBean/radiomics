#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/project/community/$(whoami)/mu-glioma"
REMOTE_LOG="/tmp/mu_glioma_progression_replication.log"
REMOTE_PID="/tmp/mu_glioma_progression_replication.pid"

ssh oc4 "cd ${REPO_ROOT} && nohup .venv/bin/python scripts/run_progression_replication.py --workers 8 > ${REMOTE_LOG} 2>&1 & echo \$! > ${REMOTE_PID} && cat ${REMOTE_PID}"
echo "remote log: ${REMOTE_LOG}"
