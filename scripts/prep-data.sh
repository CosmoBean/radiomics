#!/usr/bin/env bash
# Stop immediately if any command fails.
set -euo pipefail

# Run from the repo root instead of the scripts folder.
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

"${PYTHON_BIN}" main.py prep-data
