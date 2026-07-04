#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${OPENFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"


export STEER_VLN_SEEN_NOGS_NOGTAV_CONFIG="${STEER_VLN_SEEN_NOGS_NOGTAV_CONFIG:-dataset/Annotation/filtered/seen_ordered_no_gs_no_gtav.json}"
export STEER_VLN_EVAL_TEST_CONFIG="${STEER_VLN_SEEN_NOGS_NOGTAV_CONFIG}"
export EVAL_TEST_CONFIG="${STEER_VLN_SEEN_NOGS_NOGTAV_CONFIG}"
export EVAL_CONFIG="${STEER_VLN_SEEN_NOGS_NOGTAV_CONFIG}"
export OPENFLY_EVAL_SPLIT="test_seen"
export STEER_VLN_EVAL_SPLIT="test_seen"

LOG_ROOT="runs/STEER-VLN/logs/eval_seen_unseen_ablation_ordered_no_gs_no_gtav/seen"
METRICS_ROOT="runs/STEER-VLN/metrics/eval_seen_unseen_ablation_ordered_no_gs_no_gtav/seen"
mkdir -p "$LOG_ROOT" "$METRICS_ROOT"

timestamp() {
  date +%Y-%m-%d_%H-%M-%S
}

TAG="B0_original_openfly_baseline_seen_ordered_no_gs_no_gtav"
LOG="${LOG_ROOT}/${TAG}_$(timestamp).log"

export EVAL_METRICS_ROOT="${METRICS_ROOT}"
export EVAL_METRICS_TAG="${TAG}"
export BASELINE_METHOD="B0_original_openfly_baseline"

echo "============================================================"
echo "[B0 ONLY / SEEN / NO_GS_NO_GTAV] Original OpenFly baseline"
echo "[CONFIG] ${STEER_VLN_SEEN_NOGS_NOGTAV_CONFIG}"
echo "[LOG] ${LOG}"
echo "[METRICS_ROOT] ${EVAL_METRICS_ROOT}"
echo "[METRICS_TAG] ${EVAL_METRICS_TAG}"
echo "[WARNING] env_gs_* and GTA/GTA V samples are excluded; this is partial evaluation."
echo "============================================================"

if [[ ! -f "train/eval_baseline_stop_metrics.py" ]]; then
  echo "[FAIL] missing: train/eval_baseline_stop_metrics.py"
  exit 1
fi

python train/eval_baseline_stop_metrics.py \
  --eval_json "${STEER_VLN_SEEN_NOGS_NOGTAV_CONFIG}" \
  --eval_split "test_seen" \
  --metrics_root "${METRICS_ROOT}" \
  --metrics_tag "${TAG}" \
  --baseline_method "B0_original_openfly_baseline" \
  2>&1 | tee "${LOG}"

echo "============================================================"
echo "[DONE] B0 seen no_gs_no_gtav finished"
echo "[SAVED] ${LOG}"
echo "============================================================"

python STEER-VLN/summarize_steer_vln_final_ablation_logs.py "${LOG_ROOT}"
