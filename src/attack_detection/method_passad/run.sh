#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PV_ROOT="$(cd "${PACKAGE_ROOT}/../../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${HOME}/miniconda3/envs/pvparser/bin/python" ]]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/pvparser/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi
OUTPUT_ROOT="${PACKAGE_ROOT}/reproduced_results"
OUTPUT_PREFIX="pasad_family_vote1_s2345_reproduced"

if [ "$#" -eq 0 ]; then
  PASAD_ARGS=(--scenarios s2 s3 s4 s5)
elif [[ "${1}" == -* ]]; then
  PASAD_ARGS=("$@")
else
  PASAD_ARGS=(--scenarios "$@")
fi

mkdir -p "${OUTPUT_ROOT}"

cd "${PV_ROOT}"

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
"${PYTHON_BIN}" "${PACKAGE_ROOT}/scripts/pasad_clc_vote1_native_label_table.py" \
  "${PASAD_ARGS[@]}" \
  --force-pasad-vote 1 \
  --output-root "${OUTPUT_ROOT}" \
  --output-prefix "${OUTPUT_PREFIX}"
