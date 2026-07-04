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

export STEER_VLN_UNSEEN_NOGS_NOGTAV_CONFIG="${STEER_VLN_UNSEEN_NOGS_NOGTAV_CONFIG:-dataset/Annotation/filtered/unseen_ordered_no_gs_no_gtav.json}"
export STEER_VLN_EVAL_TEST_CONFIG="${STEER_VLN_UNSEEN_NOGS_NOGTAV_CONFIG}"
export EVAL_TEST_CONFIG="${STEER_VLN_UNSEEN_NOGS_NOGTAV_CONFIG}"
export EVAL_CONFIG="${STEER_VLN_UNSEEN_NOGS_NOGTAV_CONFIG}"
export OPENFLY_EVAL_SPLIT="test_unseen"
export STEER_VLN_EVAL_SPLIT="test_unseen"

LOG_ROOT="runs/STEER-VLN/logs/eval_seen_unseen_ablation_ordered_no_gs_no_gtav/unseen"
METRICS_ROOT="runs/STEER-VLN/metrics/eval_seen_unseen_ablation_ordered_no_gs_no_gtav/unseen"
mkdir -p "$LOG_ROOT" "$METRICS_ROOT"

timestamp() {
  date +%Y-%m-%d_%H-%M-%S
}
export STEER_VLN_FINAL_MODEL_DIR="${STEER_VLN_FINAL_MODEL_DIR:-runs/STEER-VLN/train/final_model}"
FINAL_DIR="${STEER_VLN_FINAL_MODEL_DIR}"
LORA_PATH="${FINAL_DIR}/lora_adapter_best"
KFM_PATH="${FINAL_DIR}/keyframe_scorer_best.pt"
KFM_VOCAB="${FINAL_DIR}/simple_tokenizer_vocab.json"

TAG="E2_integrated_lora_keyframe_no_trend_unseen_ordered_no_gs_no_gtav"
LOG="${LOG_ROOT}/${TAG}_$(timestamp).log"

export EVAL_METRICS_ROOT="${METRICS_ROOT}"
export EVAL_METRICS_TAG="${TAG}"
export BASELINE_METHOD="${TAG}"

echo "============================================================"
echo "[E2 ONLY / UNSEEN / NO_GS_NO_GTAV] Preflight check"
echo "[CONFIG] ${STEER_VLN_UNSEEN_NOGS_NOGTAV_CONFIG}"
echo "[LOG] ${LOG}"
echo "[METRICS_ROOT] ${EVAL_METRICS_ROOT}"
echo "[METRICS_TAG] ${EVAL_METRICS_TAG}"
echo "[WARNING] env_gs_* and GTA/GTA V samples are excluded; this is partial evaluation."
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
  AIRSIM_PORT="${AIRSIM_PORT:-41451}" \
  python STEER-VLN/eval_hd_lora.py \
    --eval_json "${STEER_VLN_UNSEEN_NOGS_NOGTAV_CONFIG}" \
    --eval_split "test_unseen" \
    --metrics_root "${METRICS_ROOT}" \
    --metrics_tag "${TAG}" \
    --baseline_method "${TAG}" \
    2>&1 | tee "${LOG}"

echo "============================================================"
echo "[DONE] E2 unseen no_gs_no_gtav finished"
echo "[SAVED] ${LOG}"
echo "============================================================"

python STEER-VLN/summarize_steer_vln_final_ablation_logs.py "${LOG_ROOT}"
