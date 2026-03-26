#!/bin/bash
set -euo pipefail

#=========================================#
# Repo roots
#=========================================#
PROJ_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY_CACHE_BASE="${PROJ_DIR}/../cache"

export PIP_CACHE_DIR="${PY_CACHE_BASE}/pip"
export UV_CACHE_DIR="${PY_CACHE_BASE}/uv"
export XDG_CACHE_HOME="${PY_CACHE_BASE}"
export HF_HOME="${PY_CACHE_BASE}/huggingface"
export HUGGINGFACE_HUB_CACHE="${PY_CACHE_BASE}/huggingface"
export TORCH_HOME="${PY_CACHE_BASE}/torch"
export WANDB_CACHE_DIR="${PY_CACHE_BASE}/wandb"
export TRITON_CACHE_DIR="${PY_CACHE_BASE}/triton"
export DATASETS_CACHE="${PY_CACHE_BASE}/datasets"
export MPLCONFIGDIR="${PY_CACHE_BASE}/matplotlib"
export HF_DATASETS_CACHE="${PY_CACHE_BASE}/datasets"
export HF_HUB_CACHE="${PY_CACHE_BASE}/huggingface"

mkdir -p \
  "${PIP_CACHE_DIR}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${WANDB_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${DATASETS_CACHE}" \
  "${MPLCONFIGDIR}"

#=========================================#
# CPU detection
#=========================================#
detect_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
    return
  fi
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.logicalcpu
    return
  fi
  echo 1
}

CPU_COUNT=$(detect_cpu_count)
if ! [[ ${CPU_COUNT} =~ ^[0-9]+$ ]] || [[ ${CPU_COUNT} -lt 1 ]]; then
  CPU_COUNT=1
fi

UV_CONCURRENT_BUILDS=$((CPU_COUNT < 8 ? CPU_COUNT : 8))
UV_CONCURRENT_DOWNLOADS=$((CPU_COUNT * 2))
if [[ ${UV_CONCURRENT_DOWNLOADS} -gt 32 ]]; then
  UV_CONCURRENT_DOWNLOADS=32
fi
UV_CONCURRENT_INSTALLS=$((CPU_COUNT < 16 ? CPU_COUNT : 16))

export UV_CONCURRENT_BUILDS
export UV_CONCURRENT_DOWNLOADS
export UV_CONCURRENT_INSTALLS

echo "Detected ${CPU_COUNT} CPU cores; uv concurrency: builds=${UV_CONCURRENT_BUILDS}, downloads=${UV_CONCURRENT_DOWNLOADS}, installs=${UV_CONCURRENT_INSTALLS}"

#=========================================#
# Bootstrap
#=========================================#
UV_BIN=$(command -v uv)
if [[ -z "${UV_BIN}" ]]; then
  echo "uv not found on PATH; install uv first" >&2
  exit 1
fi
UV_BIN_DIR=$(dirname "${UV_BIN}")
if [[ "${UV_BIN_DIR}" == "${PROJ_DIR}/.local/bin" ]]; then
  echo "Updating uv..."
  uv self update || echo "Warning: uv self-update failed"
else
  echo "Skipping uv self-update (binary ${UV_BIN} outside ${PROJ_DIR}/.local/bin)"
fi

if [[ ! -d .venv ]]; then
  uv venv --clear --python 3.11
fi

#=========================================#
# Core dependencies (pinned to recent releases)
#=========================================#
uv pip install --python .venv/bin/python numpy==2.2.6
uv pip install --python .venv/bin/python \
  pandas==2.3.3 \
  scikit-learn==1.7.2 \
  lightgbm==4.6.0 \
  optuna==4.8.0 \
  shap==0.49.1 \
  matplotlib==3.10.8 \
  nibabel==5.4.2 \
  SimpleITK==2.5.3 \
  huggingface-hub==1.7.2 \
  tqdm==4.67.3 \
  PyYAML==6.0.3 \
  gpustat==1.1.1

uv pip install --python .venv/bin/python setuptools==82.0.1 wheel==0.46.3

uv pip install --python .venv/bin/python --no-build-isolation pyradiomics==3.0.1

echo "Installation complete; activate with: source .venv/bin/activate"
