#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ "$(type -t conda)" != "function" ]; then
  __conda_base="$(conda info --base 2>/dev/null || true)"
  if [ -n "${__conda_base}" ] && [ -f "${__conda_base}/etc/profile.d/conda.sh" ]; then
    source "${__conda_base}/etc/profile.d/conda.sh"
  fi
fi
conda activate torch132
export REPO_ROOT
export TORCH132_ENV="$CONDA_PREFIX"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PYBIND_BUILD="$REPO_ROOT/flashattention/v3_mMn1k1/pybind/build"
export LD_LIBRARY_PATH="$TORCH132_ENV/lib:${LD_LIBRARY_PATH:-}"

rm -rf "$PYBIND_BUILD"

cmake -S "$REPO_ROOT/flashattention/v3_mMn1k1/pybind" -B "$PYBIND_BUILD" \
  -G Ninja \
  -DTORCH132_ENV="$TORCH132_ENV" \
  -DCMAKE_MAKE_PROGRAM="$TORCH132_ENV/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$PYBIND_BUILD" -j8 &> "$PYBIND_BUILD/build.log"

python "$SCRIPT_DIR/extract_resources.py" "$PYBIND_BUILD/build.log" "$PYBIND_BUILD/resource.log"
