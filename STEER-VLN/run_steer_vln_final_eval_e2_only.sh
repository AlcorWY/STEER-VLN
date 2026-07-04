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

export STEER_VLN_FINAL_MODEL_DIR="${STEER_VLN_FINAL_MODEL_DIR:-runs/STEER-VLN/train/final_model}"
FINAL_DIR="${STEER_VLN_FINAL_MODEL_DIR}"
LORA_PATH="${FINAL_DIR}/lora_adapter_best"
KFM_PATH="${FINAL_DIR}/keyframe_scorer_best.pt"
KFM_VOCAB="${FINAL_DIR}/simple_tokenizer_vocab.json"
TAG="E2_integrated_lora_keyframe_no_trend"
LOG="${LOG_ROOT}/${TAG}_$(timestamp).log"

export EVAL_METRICS_ROOT="${METRICS_ROOT}"
export EVAL_METRICS_TAG="${TAG}"

echo "============================================================"
echo "[E2 ONLY / MAIN] Preflight check"
echo "[CONFIG] ${STEER_VLN_MAIN_TEST_CONFIG}"
echo "[LOG] ${LOG}"
echo "[METRICS_ROOT] ${EVAL_METRICS_ROOT}"
echo "[METRICS_TAG] ${EVAL_METRICS_TAG}"
echo "============================================================"

required_files=("${KFM_PATH}" "${KFM_VOCAB}" "${FINAL_DIR}/integrated_full_best.pt")
required_dirs=("${LORA_PATH}")
for f in "${required_files[@]}"; do [[ -f "$f" ]] || { echo "[FAIL] missing file: $f"; exit 1; }; echo "[OK] $f"; done
for d in "${required_dirs[@]}"; do [[ -d "$d" ]] || { echo "[FAIL] missing dir: $d"; exit 1; }; echo "[OK] $d"; done

echo "============================================================"
echo "[RUN] ${TAG}"
echo "============================================================"

env \
  LORA_ADAPTER_PATH="${LORA_PATH}" \
  LEARNED_KEYFRAME_CHECKPOINT="${KFM_PATH}" \
  LEARNED_KEYFRAME_VOCAB="${KFM_VOCAB}" \
  KEYFRAME_MODE=learned_topk \
  LEARNED_KEYFRAME_EXCLUDE_PREVIOUS=1 \
  LEARNED_KEYFRAME_MAX_HISTORY=8 \
  LEARNED_KEYFRAME_TOPK=3 \
  TREND_CONDITIONED=0 \
  ACTION_MODE=openfly \
  AIRSIM_PORT=41451 \
  python STEER-VLN/eval_hd_lora.py 2>&1 | tee "${LOG}"

echo "============================================================"
echo "[DONE] E2 finished"
echo "[SAVED] ${LOG}"
echo "============================================================"

python STEER-VLN/summarize_steer_vln_final_ablation_logs.py "${LOG_ROOT}"
