#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export SELD_BOOT_LOG="${SELD_BOOT_LOG:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "=== $(date '+%F %T') Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
echo "=== $(date '+%F %T') Step 2: Train Baseline ==="
stdbuf -oL -eL python3 -u -X faulthandler train_seldnet.py 3 baseline3

echo "=== $(date '+%F %T') Step 3: Evaluate Baseline ==="
stdbuf -oL -eL python3 -u -X faulthandler cls_compute_seld_results.py 3 baseline3

echo "=== $(date '+%F %T') DONE ==="
