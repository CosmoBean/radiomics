#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${1:-oc4}"
OUTPUT_DIR="${2:-results/postoperative_progression_surveillance_model_selection}"
REMOTE_REPO="/project/community/$(whoami)/mu-glioma"
REMOTE_LOG="${3:-/tmp/mu_glioma_postoperative_progression_model_selection.log}"

ssh "${REMOTE_HOST}" "bash -lc '
cd ${REMOTE_REPO}
echo \"timestamp\"
date \"+%Y-%m-%d %H:%M:%S %Z\"
echo \"--- process ---\"
ps -o pid,etime,pcpu,pmem,cmd -C python | grep \"run_postoperative_progression_surveillance.py\" | grep -- \"--output-dir ${OUTPUT_DIR}\" || true
echo \"--- progress ---\"
if [ -f ${OUTPUT_DIR}/progress.json ]; then
  .venv/bin/python - <<\"PY\"
from pathlib import Path
import json
p = Path(\"${OUTPUT_DIR}/progress.json\")
obj = json.loads(p.read_text())
for key in sorted(obj):
    print(f\"{key}: {obj[key]}\")
PY
else
  echo \"progress.json not written by this run\"
fi
echo \"--- files ---\"
find ${OUTPUT_DIR} -maxdepth 1 -type f | sort
echo \"--- model_search ---\"
if [ -f ${OUTPUT_DIR}/model_search.csv ]; then
  .venv/bin/python - <<\"PY\"
from pathlib import Path
import pandas as pd
p = Path(\"${OUTPUT_DIR}/model_search.csv\")
df = pd.read_csv(p)
if df.empty:
    print(\"model_search.csv exists but is empty\")
else:
    cols = [c for c in [\"model\", \"subset_size\", \"mean_fold_auc\", \"min_fold_auc\", \"best_fold_auc\"] if c in df.columns]
    print(df.sort_values(cols[2:], ascending=False).head(10)[cols].to_string(index=False))
PY
else
  echo \"model_search.csv not written yet\"
fi
echo \"--- log_tail ---\"
tail -n 20 ${REMOTE_LOG} 2>/dev/null || true
'"
