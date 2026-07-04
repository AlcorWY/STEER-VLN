#!/usr/bin/env bash
set -euo pipefail

cd "${OPENFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OPENFLY_EVAL_SPLIT="${OPENFLY_EVAL_SPLIT:-${STEER_VLN_EVAL_SPLIT:-demo}}"

mkdir -p "runs/openfly_source/logs/${OPENFLY_EVAL_SPLIT}"
LOG="runs/openfly_source/logs/${OPENFLY_EVAL_SPLIT}/openfly_source_eval_${OPENFLY_EVAL_SPLIT}_$(date +%Y-%m-%d_%H-%M-%S).log"

echo "[OpenFly Source Eval] split=${OPENFLY_EVAL_SPLIT}"
echo "[OpenFly Source Eval] log=${LOG}"

python train/eval_baseline_stop_metrics.py 2>&1 | tee "${LOG}"
