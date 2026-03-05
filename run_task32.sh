#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[ERROR] line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK_ID="${TASK_ID:-32}"
JOB_ID="${JOB_ID:-adaptive32}"
GPU_ID="${GPU_ID:-1}"
RUN_FEAT="${RUN_FEAT:-1}"

mkdir -p logs
TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="logs/task${TASK_ID}_${JOB_ID}_${TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export SELD_BOOT_LOG="${SELD_BOOT_LOG:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"

echo "=== $(date '+%F %T') SELD 파이프라인 시작 ==="
echo "TASK_ID=${TASK_ID}, JOB_ID=${JOB_ID}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, RUN_FEAT=${RUN_FEAT}"
echo "LOG_FILE=${LOG_FILE}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "seld" ]]; then
    echo "[경고] 현재 conda env가 'seld'가 아닙니다. (현재: ${CONDA_DEFAULT_ENV:-none})"
fi

if [[ "${RUN_FEAT}" == "1" ]]; then
    echo "=== $(date '+%F %T') Step 1: Feature Extraction (task ${TASK_ID}) ==="
    stdbuf -oL -eL python3 -u -X faulthandler batch_feature_extraction.py "${TASK_ID}"
else
    echo "=== $(date '+%F %T') Step 1: Feature Extraction 건너뜀 (RUN_FEAT=${RUN_FEAT}) ==="
fi

echo "=== $(date '+%F %T') Step 2: Train + Dev-Test Eval (task ${TASK_ID}, job ${JOB_ID}) ==="
stdbuf -oL -eL python3 -u -X faulthandler train_seldnet.py "${TASK_ID}" "${JOB_ID}"

echo "=== $(date '+%F %T') 완료 ==="
echo "모델 폴더: ${SCRIPT_DIR}/models_audio"
echo "결과 폴더: ${SCRIPT_DIR}/results_audio"
