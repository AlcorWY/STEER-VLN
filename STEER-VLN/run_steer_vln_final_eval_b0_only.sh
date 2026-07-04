#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${OPENFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Main experiment only: no seen/unseen split is used.
export STEER_VLN_MAIN_TEST_CONFIG="${STEER_VLN_MAIN_TEST_CONFIG:-configs/eval_test.json}"
export STEER_VLN_EVAL_TEST_CONFIG="${STEER_VLN_MAIN_TEST_CONFIG}"
export EVAL_TEST_CONFIG="${STEER_VLN_MAIN_TEST_CONFIG}"

LOG_ROOT="runs/STEER-VLN/logs/eval_main_ablation"
METRICS_ROOT="runs/STEER-VLN/metrics/eval_main_ablation"
mkdir -p "$LOG_ROOT" "$METRICS_ROOT"

timestamp() {
  date +%Y-%m-%d_%H-%M-%S
}

TAG="B0_original_openfly_baseline"
LOG="${LOG_ROOT}/${TAG}_$(timestamp).log"

export EVAL_METRICS_ROOT="${METRICS_ROOT}"
export EVAL_METRICS_TAG="${TAG}"

echo "============================================================"
echo "[B0 ONLY / MAIN] Original OpenFly baseline"
echo "[CONFIG] ${STEER_VLN_MAIN_TEST_CONFIG}"
echo "[LOG] ${LOG}"
echo "[METRICS_ROOT] ${EVAL_METRICS_ROOT}"
echo "[METRICS_TAG] ${EVAL_METRICS_TAG}"
echo "============================================================"

if [[ ! -f "train/eval_baseline_stop_metrics.py" ]]; then
  echo "[FAIL] missing: train/eval_baseline_stop_metrics.py"
  exit 1
fi

python train/eval_baseline_stop_metrics.py 2>&1 | tee "${LOG}"

echo "============================================================"
echo "[DONE] B0 finished"
echo "[SAVED] ${LOG}"
echo "============================================================"

python STEER-VLN/summarize_steer_vln_final_ablation_logs.py "${LOG_ROOT}"
