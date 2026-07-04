#!/usr/bin/env bash
set -euo pipefail

cd "${OPENFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

for split in test_seen test_unseen; do
  echo "============================================================"
  echo "[OpenFly Source Eval] ${split}"
  echo "============================================================"
  OPENFLY_EVAL_SPLIT="${split}" bash train/run_openfly_eval_split.sh
done
