from pathlib import Path

ROOT = Path("STEER-VLN")
ROOT.mkdir(parents=True, exist_ok=True)

COMMON = r'''#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${OPENFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FINAL_DIR="${STEER_VLN_FINAL_MODEL_DIR:-runs/STEER-VLN/train/final_model}"
LOG_BASE="runs/STEER-VLN/logs/eval_final_ablation"

timestamp() {
  date +%Y-%m-%d_%H-%M-%S
}

resolve_split_config() {
  local split="$1"

  if [[ "$split" == "test_seen" ]]; then
    if [[ -n "${STEER_VLN_TEST_SEEN_CONFIG:-}" && -f "${STEER_VLN_TEST_SEEN_CONFIG}" ]]; then
      echo "${STEER_VLN_TEST_SEEN_CONFIG}"
      return 0
    fi

    local candidates=(
      "configs/eval_test_seen.json"
      "configs/eval_seen.json"
      "configs/test_seen.json"
      "dataset/Annotation/eval_test_seen.json"
      "dataset/Annotation/eval_seen.json"
      "dataset/Annotation/test_seen.json"
      "dataset/Annotation/test-seen.json"
    )

    for c in "${candidates[@]}"; do
      if [[ -f "$c" ]]; then
        echo "$c"
        return 0
      fi
    done

    echo "[FAIL] cannot find test_seen config." >&2
    echo "Set manually:" >&2
    echo "  STEER_VLN_TEST_SEEN_CONFIG=dataset/Annotation/your_seen.json bash $0" >&2
    exit 1
  fi

  if [[ "$split" == "test_unseen" ]]; then
    if [[ -n "${STEER_VLN_TEST_UNSEEN_CONFIG:-}" && -f "${STEER_VLN_TEST_UNSEEN_CONFIG}" ]]; then
      echo "${STEER_VLN_TEST_UNSEEN_CONFIG}"
      return 0
    fi

    local candidates=(
      "configs/eval_test_unseen.json"
      "configs/eval_unseen.json"
      "configs/test_unseen.json"
      "dataset/Annotation/eval_test_unseen.json"
      "dataset/Annotation/eval_unseen.json"
      "dataset/Annotation/test_unseen.json"
      "dataset/Annotation/test-unseen.json"
    )

    for c in "${candidates[@]}"; do
      if [[ -f "$c" ]]; then
        echo "$c"
        return 0
      fi
    done

    echo "[FAIL] cannot find test_unseen config." >&2
    echo "Set manually:" >&2
    echo "  STEER_VLN_TEST_UNSEEN_CONFIG=dataset/Annotation/your_unseen.json bash $0" >&2
    exit 1
  fi

  echo "[FAIL] unknown split: $split" >&2
  exit 1
}

summarize_split() {
  local split="$1"
  local log_dir="${LOG_BASE}/${split}"

  if [[ -f STEER-VLN/summarize_steer_vln_final_ablation_logs.py ]]; then
    python STEER-VLN/summarize_steer_vln_final_ablation_logs.py "$log_dir" || true
  fi
}

require_final_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[FAIL] missing file: $f"
    echo "Please run training first:"
    echo "  bash STEER-VLN/run_steer_vln_final_train.sh"
    exit 1
  fi
}

require_final_dir() {
  local d="$1"
  if [[ ! -d "$d" ]]; then
    echo "[FAIL] missing dir: $d"
    echo "Please run training first:"
    echo "  bash STEER-VLN/run_steer_vln_final_train.sh"
    exit 1
  fi
}
'''

def write(name: str, body: str):
    path = ROOT / name
    path.write_text(COMMON + "\n" + body, encoding="utf-8")
    path.chmod(0o755)
    print(f"[WRITE] {path}")


# ============================================================
# B0: Original OpenFly baseline
# ============================================================

write("run_steer_vln_b0_seen_unseen.sh", r'''
CASE_TAG="B0_original_openfly_baseline"

run_one_split() {
  local split="$1"
  local cfg
  cfg="$(resolve_split_config "$split")"

  local log_dir="${LOG_BASE}/${split}"
  mkdir -p "$log_dir"

  local log="${log_dir}/${CASE_TAG}_$(timestamp).log"

  echo ""
  echo "============================================================"
  echo "[STEER-VLN][${CASE_TAG}] split=${split}"
  echo "[CONFIG] ${cfg}"
  echo "[LOG] ${log}"
  echo "============================================================"

  env \
    STEER_VLN_EVAL_SPLIT="${split}" \
    OPENFLY_EVAL_SPLIT="${split}" \
    EVAL_CONFIG="${cfg}" \
    BASELINE_METHOD="${CASE_TAG}" \
    python train/eval_baseline_stop_metrics.py \
    2>&1 | tee "$log"

  summarize_split "$split"
}

run_one_split test_seen
run_one_split test_unseen

echo "[DONE] ${CASE_TAG} seen+unseen finished."
''')


# ============================================================
# E1: LoRA only, residual keyframe, no trend
# ============================================================

write("run_steer_vln_e1_seen_unseen.sh", r'''
CASE_TAG="E1_integrated_lora_only_residual_no_trend"

require_final_dir  "${FINAL_DIR}/lora_adapter_best"
require_final_file "${FINAL_DIR}/m3c_trend_head_best.pt"

run_one_split() {
  local split="$1"
  local cfg
  cfg="$(resolve_split_config "$split")"

  local log_dir="${LOG_BASE}/${split}"
  mkdir -p "$log_dir"

  local log="${log_dir}/${CASE_TAG}_$(timestamp).log"

  echo ""
  echo "============================================================"
  echo "[STEER-VLN][${CASE_TAG}] split=${split}"
  echo "[CONFIG] ${cfg}"
  echo "[FINAL_DIR] ${FINAL_DIR}"
  echo "[LOG] ${log}"
  echo "============================================================"

  env \
    STEER_VLN_EVAL_SPLIT="${split}" \
    OPENFLY_EVAL_SPLIT="${split}" \
    EVAL_CONFIG="${cfg}" \
    LORA_ADAPTER_PATH="${FINAL_DIR}/lora_adapter_best" \
    M3C_TREND_HEAD_CHECKPOINT="${FINAL_DIR}/m3c_trend_head_best.pt" \
    KEYFRAME_MODE=residual \
    TREND_CONDITIONED=0 \
    ACTION_MODE=openfly \
    python STEER-VLN/eval_hd_lora.py \
    2>&1 | tee "$log"

  summarize_split "$split"
}

run_one_split test_seen
run_one_split test_unseen

echo "[DONE] ${CASE_TAG} seen+unseen finished."
''')


# ============================================================
# E2: LoRA + jointly-trained keyframe, no trend
# ============================================================

write("run_steer_vln_e2_seen_unseen.sh", r'''
CASE_TAG="E2_integrated_lora_keyframe_no_trend"

require_final_dir  "${FINAL_DIR}/lora_adapter_best"
require_final_file "${FINAL_DIR}/m3c_trend_head_best.pt"
require_final_file "${FINAL_DIR}/keyframe_scorer_best.pt"
require_final_file "${FINAL_DIR}/simple_tokenizer_vocab.json"

run_one_split() {
  local split="$1"
  local cfg
  cfg="$(resolve_split_config "$split")"

  local log_dir="${LOG_BASE}/${split}"
  mkdir -p "$log_dir"

  local log="${log_dir}/${CASE_TAG}_$(timestamp).log"

  echo ""
  echo "============================================================"
  echo "[STEER-VLN][${CASE_TAG}] split=${split}"
  echo "[CONFIG] ${cfg}"
  echo "[FINAL_DIR] ${FINAL_DIR}"
  echo "[LOG] ${log}"
  echo "============================================================"

  env \
    STEER_VLN_EVAL_SPLIT="${split}" \
    OPENFLY_EVAL_SPLIT="${split}" \
    EVAL_CONFIG="${cfg}" \
    LORA_ADAPTER_PATH="${FINAL_DIR}/lora_adapter_best" \
    M3C_TREND_HEAD_CHECKPOINT="${FINAL_DIR}/m3c_trend_head_best.pt" \
    LEARNED_KEYFRAME_CHECKPOINT="${FINAL_DIR}/keyframe_scorer_best.pt" \
    LEARNED_KEYFRAME_VOCAB="${FINAL_DIR}/simple_tokenizer_vocab.json" \
    KEYFRAME_MODE=learned_topk \
    LEARNED_KEYFRAME_EXCLUDE_PREVIOUS=1 \
    LEARNED_KEYFRAME_MAX_HISTORY=8 \
    LEARNED_KEYFRAME_TOPK=3 \
    TREND_CONDITIONED=0 \
    ACTION_MODE=openfly \
    python STEER-VLN/eval_hd_lora.py \
    2>&1 | tee "$log"

  summarize_split "$split"
}

run_one_split test_seen
run_one_split test_unseen

echo "[DONE] ${CASE_TAG} seen+unseen finished."
''')


# ============================================================
# E3: Final full model, LoRA + keyframe + temporal trend
# ============================================================

write("run_steer_vln_e3_seen_unseen.sh", r'''
CASE_TAG="E3_final_integrated_lora_keyframe_temporal_trend"

require_final_dir  "${FINAL_DIR}/lora_adapter_best"
require_final_file "${FINAL_DIR}/m3c_trend_head_best.pt"
require_final_file "${FINAL_DIR}/keyframe_scorer_best.pt"
require_final_file "${FINAL_DIR}/simple_tokenizer_vocab.json"

run_one_split() {
  local split="$1"
  local cfg
  cfg="$(resolve_split_config "$split")"

  local log_dir="${LOG_BASE}/${split}"
  mkdir -p "$log_dir"

  local log="${log_dir}/${CASE_TAG}_$(timestamp).log"

  echo ""
  echo "============================================================"
  echo "[STEER-VLN][${CASE_TAG}] split=${split}"
  echo "[CONFIG] ${cfg}"
  echo "[FINAL_DIR] ${FINAL_DIR}"
  echo "[LOG] ${log}"
  echo "============================================================"

  env \
    STEER_VLN_EVAL_SPLIT="${split}" \
    OPENFLY_EVAL_SPLIT="${split}" \
    EVAL_CONFIG="${cfg}" \
    LORA_ADAPTER_PATH="${FINAL_DIR}/lora_adapter_best" \
    M3C_TREND_HEAD_CHECKPOINT="${FINAL_DIR}/m3c_trend_head_best.pt" \
    LEARNED_KEYFRAME_CHECKPOINT="${FINAL_DIR}/keyframe_scorer_best.pt" \
    LEARNED_KEYFRAME_VOCAB="${FINAL_DIR}/simple_tokenizer_vocab.json" \
    KEYFRAME_MODE=learned_topk \
    LEARNED_KEYFRAME_EXCLUDE_PREVIOUS=1 \
    LEARNED_KEYFRAME_MAX_HISTORY=8 \
    LEARNED_KEYFRAME_TOPK=3 \
    TREND_CONDITIONED=1 \
    TREND_STATE_TEMPORAL_WINDOW=3 \
    TREND_STATE_STOP_THRESHOLD=0.50 \
    TREND_STATE_PRESTOP_THRESHOLD=0.50 \
    TREND_STATE_DEBUG=1 \
    ACTION_MODE=openfly \
    WAYPOINT_FORWARD_POLICY=distance_bins \
    WAYPOINT_USE_Z_DECODE=1 \
    python STEER-VLN/eval_hd_lora.py \
    2>&1 | tee "$log"

  summarize_split "$split"
}

run_one_split test_seen
run_one_split test_unseen

echo "[DONE] ${CASE_TAG} seen+unseen finished."
''')


# ============================================================
# Optional: run all four independent scripts
# ============================================================

write("run_steer_vln_all_b0_e1_e2_e3_seen_unseen.sh", r'''
echo "============================================================"
echo "[STEER-VLN] Run B0/E1/E2/E3 as separated scripts on seen+unseen"
echo "============================================================"

bash STEER-VLN/run_steer_vln_b0_seen_unseen.sh
bash STEER-VLN/run_steer_vln_e1_seen_unseen.sh
bash STEER-VLN/run_steer_vln_e2_seen_unseen.sh
bash STEER-VLN/run_steer_vln_e3_seen_unseen.sh

echo "============================================================"
echo "[DONE] All separated STEER_VLN seen+unseen experiments finished."
echo "============================================================"
''')

print("[DONE] Created separated B0/E1/E2/E3 seen+unseen scripts.")
