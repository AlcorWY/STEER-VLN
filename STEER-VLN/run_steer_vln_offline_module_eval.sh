#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${OPENFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FINAL_DIR="${STEER_VLN_FINAL_MODEL_DIR:-runs/STEER-VLN/train/final_model}"
MODEL_PATH="${MODEL_PATH:-models/openfly-agent-7b}"
PARQUET_ROOT="${PARQUET_ROOT:-dataset/hf_openfly_airsim16/traj}"
OUT_ROOT="${OUT_ROOT:-runs/STEER-VLN/offline_module_eval}"

# 离线模块评估只跑完整 eval_test，不再跑 seen/unseen
EVAL_TEST_JSON="${STEER_VLN_EVAL_TEST_CONFIG:-configs/eval_test.json}"

# 0 表示完整评估，不截断
KEYFRAME_MAX="${STEER_VLN_OFFLINE_MAX_SAMPLES_KEYFRAME:-0}"
INTEGRATED_MAX="${STEER_VLN_OFFLINE_MAX_SAMPLES_INTEGRATED:-0}"
VIS_MAX_CASES="${STEER_VLN_OFFLINE_VIS_MAX_CASES:-32}"
POLICY_MAX="${STEER_VLN_OFFLINE_MAX_SAMPLES_POLICY:-0}"
POLICY_VIS_MAX="${STEER_VLN_OFFLINE_POLICY_VIS_MAX_CASES:-24}"
POLICY_MODES="${STEER_VLN_OFFLINE_POLICY_MODES:-residual_no_trend,learned_no_trend,learned_oracle_trend}"

mkdir -p "${OUT_ROOT}"

echo "============================================================"
echo "[STEER-VLN OFFLINE MODULE EVAL] Preflight"
echo "============================================================"
echo "[FINAL_DIR] ${FINAL_DIR}"
echo "[EVAL_TEST] ${EVAL_TEST_JSON}"
echo "[PARQUET] ${PARQUET_ROOT}"
echo "[MODEL_PATH] ${MODEL_PATH}"
echo "[OUT_ROOT] ${OUT_ROOT}"
echo "[KEYFRAME_MAX] ${KEYFRAME_MAX}"
echo "[INTEGRATED_MAX] ${INTEGRATED_MAX}"
echo "[VIS_MAX_CASES] ${VIS_MAX_CASES}"

required_files=(
  "STEER-VLN/check_steer_vln_scheme.py"
  "STEER-VLN/offline_eval_keyframe_module.py"
  "STEER-VLN/offline_eval_integrated_modules.py"
  "STEER-VLN/summarize_steer_vln_offline_module_eval.py"
  "${EVAL_TEST_JSON}"
  "${FINAL_DIR}/keyframe_scorer_best.pt"
  "${FINAL_DIR}/simple_tokenizer_vocab.json"
  "${FINAL_DIR}/m3c_trend_head_best.pt"
  "${FINAL_DIR}/integrated_full_best.pt"
  "${FINAL_DIR}/final_model_manifest.json"
)

required_dirs=(
  "${FINAL_DIR}/lora_adapter_best"
)

for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[FAIL] missing file: $f"
    exit 1
  fi
  echo "[OK] $f"
done

for d in "${required_dirs[@]}"; do
  if [[ ! -d "$d" ]]; then
    echo "[FAIL] missing dir: $d"
    exit 1
  fi
  echo "[OK] $d"
done

echo "============================================================"
echo "[0/5] Static scheme check"
echo "============================================================"

python STEER-VLN/check_steer_vln_scheme.py \
  --steer_vln_dir STEER-VLN \
  --final_model_dir "${FINAL_DIR}" \
  --output_dir "${OUT_ROOT}/check_scheme" || true

echo "============================================================"
echo "[1/5] Offline keyframe eval: eval_test"
echo "============================================================"

python STEER-VLN/offline_eval_keyframe_module.py \
  --final_model_dir "${FINAL_DIR}" \
  --annotation_path "${EVAL_TEST_JSON}" \
  --parquet_root "${PARQUET_ROOT}" \
  --output_dir "${OUT_ROOT}/keyframe/eval_test" \
  --max_samples "${KEYFRAME_MAX}"

echo "============================================================"
echo "[2/5] Offline integrated eval: eval_test"
echo "============================================================"

python STEER-VLN/offline_eval_integrated_modules.py \
  --final_model_dir "${FINAL_DIR}" \
  --model_path "${MODEL_PATH}" \
  --annotation_path "${EVAL_TEST_JSON}" \
  --parquet_root "${PARQUET_ROOT}" \
  --output_dir "${OUT_ROOT}/integrated/eval_test" \
  --max_samples "${INTEGRATED_MAX}" \
  --bf16


echo "============================================================"
echo "[3/6] Offline final policy action eval: eval_test"
echo "============================================================"

python STEER-VLN/offline_eval_policy_action_module.py \
  --final_model_dir "${FINAL_DIR}" \
  --model_path "${MODEL_PATH}" \
  --annotation_path "${EVAL_TEST_JSON}" \
  --parquet_root "${PARQUET_ROOT}" \
  --output_dir "${OUT_ROOT}/policy/eval_test" \
  --modes "${POLICY_MODES}" \
  --max_samples "${POLICY_MAX}" \
  --max_visual_cases "${POLICY_VIS_MAX}" \
  --bf16 || {
    echo "[WARN] final policy offline eval failed, continue."
  }

echo "============================================================"
echo "[3/5] Offline categorized module visualizations: eval_test"
echo "============================================================"

python STEER-VLN/offline_visualize_by_module.py \
  --final_model_dir "${FINAL_DIR}" \
  --annotation_path "${EVAL_TEST_JSON}" \
  --parquet_root "${PARQUET_ROOT}" \
  --output_dir "${OUT_ROOT}/visual_by_module/eval_test" \
  --split_name eval_test \
  --integrated_per_sample "${OUT_ROOT}/integrated/eval_test/per_sample.csv" \
  --max_cases "${VIS_MAX_CASES}" || {
    echo "[WARN] categorized module visualization failed, continue summarizing."
  }


echo "============================================================"
echo "[4/6] Offline integrated prediction visualizations: eval_test"
echo "============================================================"

python STEER-VLN/offline_visualize_integrated_predictions.py \
  --final_model_dir "${FINAL_DIR}" \
  --annotation_path "${EVAL_TEST_JSON}" \
  --parquet_root "${PARQUET_ROOT}" \
  --integrated_per_sample "${OUT_ROOT}/integrated/eval_test/per_sample.csv" \
  --output_dir "${OUT_ROOT}/visual_by_module/eval_test/integrated_predictions" \
  --split_name eval_test \
  --max_cases "${VIS_MAX_CASES}" || {
    echo "[WARN] integrated prediction visualization failed, continue."
  }

echo "============================================================"
echo "[4/5] Summarize offline module eval"
echo "============================================================"

python STEER-VLN/summarize_steer_vln_offline_module_eval.py --input_root "${OUT_ROOT}"

echo "============================================================"
echo "[DONE] STEER_VLN offline module eval finished."
echo "Output: ${OUT_ROOT}"
echo "============================================================"
