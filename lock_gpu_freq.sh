#!/usr/bin/env bash
set -euo pipefail

# Lock GPU clocks to the highest supported memory and graphics clocks.
#
# Usage:
#   ./lock_gpu_freq.sh        # lock GPU 0
#   ./lock_gpu_freq.sh 1      # lock GPU 1
#
# To cancel / reset locked clocks:
#   sudo nvidia-smi -i <gpu_id> -rgc   # reset locked graphics clocks
#   sudo nvidia-smi -i <gpu_id> -rmc   # reset locked memory clocks
#   sudo nvidia-smi -i <gpu_id> -rac   # reset application clocks, if any were set
#   sudo nvidia-smi -i <gpu_id> -pm 0  # optionally disable persistence mode
#
# Example reset for GPU 0:
#   sudo nvidia-smi -i 0 -rgc
#   sudo nvidia-smi -i 0 -rmc
#   sudo nvidia-smi -i 0 -rac
#   sudo nvidia-smi -i 0 -pm 0

GPU_ID="${1:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found in PATH." >&2
  exit 1
fi

SUDO=()
if [[ "${EUID}" -ne 0 ]]; then
  SUDO=(sudo)
fi

supported_clocks="$(nvidia-smi -i "${GPU_ID}" --query-supported-clocks=mem,gr --format=csv,noheader,nounits)"

max_mem_clock="$(
  awk -F, '
    {
      gsub(/[[:space:]]/, "", $1);
      if ($1 + 0 > max) max = $1 + 0;
    }
    END { if (max > 0) print max; }
  ' <<< "${supported_clocks}"
)"

max_graphics_clock="$(
  awk -F, '
    {
      gsub(/[[:space:]]/, "", $2);
      if ($2 + 0 > max) max = $2 + 0;
    }
    END { if (max > 0) print max; }
  ' <<< "${supported_clocks}"
)"

if [[ -z "${max_mem_clock}" || -z "${max_graphics_clock}" ]]; then
  echo "Failed to query supported GPU clocks for GPU ${GPU_ID}." >&2
  echo "Raw nvidia-smi output:" >&2
  echo "${supported_clocks}" >&2
  exit 1
fi

echo "GPU ${GPU_ID}: locking memory clock to ${max_mem_clock} MHz"
echo "GPU ${GPU_ID}: locking graphics clock to ${max_graphics_clock} MHz"

"${SUDO[@]}" nvidia-smi -i "${GPU_ID}" -pm 1
"${SUDO[@]}" nvidia-smi -i "${GPU_ID}" -lmc "${max_mem_clock},${max_mem_clock}"
"${SUDO[@]}" nvidia-smi -i "${GPU_ID}" -lgc "${max_graphics_clock},${max_graphics_clock}"

nvidia-smi -i "${GPU_ID}" \
  --query-gpu=index,name,persistence_mode,clocks.max.memory,clocks.max.graphics,clocks.current.memory,clocks.current.graphics \
  --format=csv
