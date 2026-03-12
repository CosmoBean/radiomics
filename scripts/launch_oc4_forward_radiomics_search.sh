#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${1:-oc4}"
PROFILE="${2:-surveillance}"
MODALITIES="${3:-t1c,flair}"
SEARCH_PRESET="${4:-full}"
REMOTE_REPO="/project/community/$(whoami)/mu-glioma"

case "${PROFILE}" in
  surveillance)
    BASE_OUTPUT_DIR="results/postoperative_progression_forward_prediction"
    FAST_OUTPUT_DIR="results/postoperative_progression_forward_fast"
    EXTRA_FLAGS="--pre-progression-only --exclude-after-late-treatment"
    ;;
  baseline)
    BASE_OUTPUT_DIR="results/postoperative_progression_baseline_prediction"
    FAST_OUTPUT_DIR="results/postoperative_progression_baseline_fast"
    EXTRA_FLAGS="--pre-progression-only --earliest-scan-only --exclude-after-late-treatment"
    ;;
  *)
    echo "Unsupported profile: ${PROFILE}" >&2
    echo "Use 'surveillance' or 'baseline'." >&2
    exit 1
    ;;
esac

OUTPUT_DIR="${BASE_OUTPUT_DIR}"

case "${SEARCH_PRESET}" in
  full)
    SEARCH_FLAGS="--models logreg,svm --feature-subsets 128,64,32,16 --n-trials 200 --ranking-folds 5 --cv-folds 5"
    ;;
  fast)
    OUTPUT_DIR="${FAST_OUTPUT_DIR}"
    SEARCH_FLAGS="--models logreg --feature-subsets 32 --n-trials 40 --ranking-folds 3 --cv-folds 3 --permutation-repeats 20 --bootstrap-iterations 200"
    ;;
  *)
    echo "Unsupported search preset: ${SEARCH_PRESET}" >&2
    echo "Use 'full' or 'fast'." >&2
    exit 1
    ;;
esac

REMOTE_LOG="/tmp/mu_glioma_${PROFILE}_forward_radiomics.log"
REMOTE_PID="/tmp/mu_glioma_${PROFILE}_forward_radiomics.pid"

ssh "${REMOTE_HOST}" "bash -lc '
cd ${REMOTE_REPO}
mkdir -p ${OUTPUT_DIR}
if [ -f ${REMOTE_PID} ] && kill -0 \$(cat ${REMOTE_PID}) 2>/dev/null; then
  kill -INT \$(cat ${REMOTE_PID}) || true
  sleep 2
fi
TEST_PATIENT_FLAG=\"\"
if [ \"${SEARCH_PRESET}\" = \"fast\" ] && [ -f ${BASE_OUTPUT_DIR}/test_patients.txt ]; then
  TEST_PATIENT_FLAG=\"--test-patients-file ${BASE_OUTPUT_DIR}/test_patients.txt\"
fi
nohup .venv/bin/python scripts/run_postoperative_progression_surveillance.py \
  --feature-table results/postoperative_progression_surveillance/radiomics_features.csv \
  --output-dir ${OUTPUT_DIR} \
  --label-mode within_window \
  --progression-window-days 120 \
  --modalities ${MODALITIES} \
  ${EXTRA_FLAGS} \
  ${SEARCH_FLAGS} \
  \${TEST_PATIENT_FLAG} \
  > ${REMOTE_LOG} 2>&1 < /dev/null &
echo \$! > ${REMOTE_PID}
cat ${REMOTE_PID}
'"

echo "Remote log: ${REMOTE_LOG}"
echo "Remote pid file: ${REMOTE_PID}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Search preset: ${SEARCH_PRESET}"
