#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${OPENFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ANNOTATION_PATH="${STEER_VLN_TRAIN_CONFIG:-dataset/Annotation/train_airsim16.json}"
PARQUET_ROOT="${STEER_VLN_PARQUET_ROOT:-dataset/hf_openfly_airsim16/traj}"
MODEL_PATH="${OPENFLY_MODEL_DIR:-models/openfly-agent-7b}"
OUTPUT_DIR="${STEER_VLN_OUTPUT_DIR:-runs/STEER-VLN/train/integrated_full}"
FINAL_DIR="${STEER_VLN_FINAL_MODEL_DIR:-runs/STEER-VLN/train/final_model}"

mkdir -p "${OUTPUT_DIR}" "${FINAL_DIR}" runs/STEER-VLN/logs/train

echo "============================================================"
echo "[TRAIN] STEER-VLN Integrated Full Model, Version-A"
echo "============================================================"

python STEER-VLN/train_integrated_full.py --overwrite \
  --annotation_path "${ANNOTATION_PATH}" \
  --parquet_root "${PARQUET_ROOT}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --final_model_dir "${FINAL_DIR}" \
  --max_episodes 0 \
  --max_history 8 \
  --image_size 224 \
  --horizon 4 \
  --stride 1 \
  --batch_size 1 \
  --epochs 1 \
  --gradient_accumulation_steps 16 \
  --val_ratio 0.001 \
  --cache_size 64 \
  --num_workers 0 \
  --learning_rate 3e-5 \
  --head_learning_rate 3e-5 \
  --kfm_learning_rate 2e-4 \
  --kfm_loss_weight 0.2 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --save_interval 20000 \
  --gradient_checkpointing \
  --bf16 \
  2>&1 | tee runs/STEER-VLN/logs/train/train_steer_vln_integrated_full_$(date +%Y-%m-%d_%H-%M-%S).log

echo "============================================================"
echo "[CHECK] final_model outputs"
echo "============================================================"

required_files=(
  "${FINAL_DIR}/keyframe_scorer_best.pt"
  "${FINAL_DIR}/simple_tokenizer_vocab.json"
  "${FINAL_DIR}/m3c_trend_head_best.pt"
  "${FINAL_DIR}/integrated_full_best.pt"
  "${FINAL_DIR}/final_model_manifest.json"
  "${FINAL_DIR}/README.txt"
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
echo "[DONE] Training finished."
echo "Final model saved at: ${FINAL_DIR}"
echo "============================================================"
