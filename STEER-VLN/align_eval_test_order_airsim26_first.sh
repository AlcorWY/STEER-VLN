#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${OPENFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"

if [[ ! -f configs/eval_test.json ]]; then
  echo "[FAIL] missing configs/eval_test.json"
  exit 1
fi

python STEER-VLN/make_openfly_ordered_no_gs_no_gtav_eval_json.py \
  --eval_test configs/eval_test.json \
  --only eval_test \
  --out_dir dataset/Annotation/filtered \
  --rewrite_eval_test

echo "[OK] configs/eval_test.json has been backed up and rewritten."
echo "[OK] Backup: configs/eval_test.json.bak"
echo "[OK] Ordered copy: dataset/Annotation/filtered/eval_test_ordered_no_gs_no_gtav.json"
