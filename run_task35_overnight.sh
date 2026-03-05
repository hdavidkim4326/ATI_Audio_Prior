#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[ERROR] line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK_ID=35
JOB_ID="${JOB_ID:-ivresdyn35}"
GPU_ID="${GPU_ID:-1}"
RUN_FEAT="${RUN_FEAT:-auto}" # auto | 1 | 0

mkdir -p logs
TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="logs/task35_${JOB_ID}_${TS}.log"
SUMMARY_FILE="logs/task35_${JOB_ID}_${TS}_summary.txt"

exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export SELD_BOOT_LOG="${SELD_BOOT_LOG:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"

echo "=== $(date '+%F %T') Task35(IV-Residual Dynamic + Early Stop) 시작 ==="
echo "TASK_ID=${TASK_ID}, JOB_ID=${JOB_ID}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, RUN_FEAT=${RUN_FEAT}"
echo "LOG_FILE=${LOG_FILE}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "seld" ]]; then
    echo "[경고] conda env가 seld가 아닙니다. (현재: ${CONDA_DEFAULT_ENV:-none})"
fi

IVRESDYN_NORM_DIR="../DCASE2024_SELD_dataset/seld_feat_label/foa_dev_norm_l2foa_ivresdyn"

DO_FEAT=1
if [[ "${RUN_FEAT}" == "0" ]]; then
    DO_FEAT=0
elif [[ "${RUN_FEAT}" == "auto" ]]; then
    if [[ -d "${IVRESDYN_NORM_DIR}" ]] && find "${IVRESDYN_NORM_DIR}" -name '*.npy' -print -quit | grep -q .; then
        DO_FEAT=0
        echo "IV-residual dynamic 정규화 feature가 이미 존재하여 Step1을 생략합니다: ${IVRESDYN_NORM_DIR}"
    fi
fi

if [[ "${DO_FEAT}" == "1" ]]; then
    echo "=== $(date '+%F %T') Step 1: Feature Extraction (task 35) ==="
    stdbuf -oL -eL python3 -u -X faulthandler batch_feature_extraction.py "${TASK_ID}"
else
    echo "=== $(date '+%F %T') Step 1: Feature Extraction 생략 ==="
fi

echo "=== $(date '+%F %T') Step 2: Train + Test Eval (task 35, job ${JOB_ID}) ==="
stdbuf -oL -eL python3 -u -X faulthandler train_seldnet.py "${TASK_ID}" "${JOB_ID}"

LATEST_RESULT_DIR="$(grep -E 'Dumping recording-wise test results in:' "${LOG_FILE}" | tail -n1 | sed 's/.*in: //')"

{
    echo "=== Task35 요약 ==="
    echo "완료시각: $(date '+%F %T')"
    echo "로그: ${LOG_FILE}"
    echo "결과폴더: ${LATEST_RESULT_DIR:-N/A}"
    echo "디버그 플롯: ${SCRIPT_DIR}/l2_adaptive_mask_debug.png"
    echo
    echo "[최종 지표]"
    grep -E "SELD score \\(early stopping metric\\)|SED metrics: F-score|DOA metrics: Angular error|Distance metrics:|Relative Distance metrics:" "${LOG_FILE}" | tail -n 5 || true
    echo
    echo "[모델 파일 후보]"
    ls -1t models_audio/*"${JOB_ID}"*_model.h5 2>/dev/null | head -n 3 || true
} | tee "${SUMMARY_FILE}"

echo "=== $(date '+%F %T') 완료 ==="
echo "SUMMARY_FILE=${SUMMARY_FILE}"
