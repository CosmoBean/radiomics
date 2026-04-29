#!/usr/bin/env bash
# Stop immediately if any command fails.
set -euo pipefail

# Run from the repo root instead of the scripts folder.
cd "$(dirname "$0")/.."

python3 main.py train
