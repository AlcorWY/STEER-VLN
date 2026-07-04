#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${OPENFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"

SRC_MAKER="STEER-VLN/make_openfly_ordered_no_gs_no_gtav_eval_json.py"
[[ -f "$SRC_MAKER" ]] || { echo "[FAIL] missing $SRC_MAKER"; exit 1; }

echo "============================================================"
echo "[1] Generate ordered no-GS-no-GTAV json"
echo "============================================================"
if [[ -f configs/eval_test.json ]]; then
  python "$SRC_MAKER"     --seen dataset/Annotation/seen.json     --unseen dataset/Annotation/unseen.json     --eval_test configs/eval_test.json     --out_dir dataset/Annotation/filtered
else
  python "$SRC_MAKER"     --seen dataset/Annotation/seen.json     --unseen dataset/Annotation/unseen.json     --out_dir dataset/Annotation/filtered
fi

echo "============================================================"
echo "[2] Create ordered copies from existing no_gs_no_gtav scripts"
echo "============================================================"

shopt -s nullglob

BASE_SCRIPTS=(
  STEER-VLN/run_steer_vln_final_eval_b0_seen_no_gs_no_gtav.sh
  STEER-VLN/run_steer_vln_final_eval_b0_unseen_no_gs_no_gtav.sh
  STEER-VLN/run_steer_vln_final_eval_e1_seen_no_gs_no_gtav.sh
  STEER-VLN/run_steer_vln_final_eval_e1_unseen_no_gs_no_gtav.sh
  STEER-VLN/run_steer_vln_final_eval_e2_seen_no_gs_no_gtav.sh
  STEER-VLN/run_steer_vln_final_eval_e2_unseen_no_gs_no_gtav.sh
  STEER-VLN/run_steer_vln_final_eval_e3_seen_no_gs_no_gtav.sh
  STEER-VLN/run_steer_vln_final_eval_e3_unseen_no_gs_no_gtav.sh
)

for src in "${BASE_SCRIPTS[@]}"; do
  if [[ ! -f "$src" ]]; then
    echo "[WARN] missing existing script, skip: $src"
    continue
  fi

  dst="${src/_no_gs_no_gtav.sh/_ordered_no_gs_no_gtav.sh}"
  cp "$src" "$dst"

  sed -i \
    -e 's/make_openfly_no_gs_no_gtav_eval_json.py/make_openfly_ordered_no_gs_no_gtav_eval_json.py/g' \
    -e 's/make_openfly_no_gs_no_gtav_eval_json.py/make_openfly_ordered_no_gs_no_gtav_eval_json.py/g' \
    -e 's/seen_no_gs_no_gtav.json/seen_ordered_no_gs_no_gtav.json/g' \
    -e 's/unseen_no_gs_no_gtav.json/unseen_ordered_no_gs_no_gtav.json/g' \
    -e 's/eval_seen_unseen_ablation_no_gs_no_gtav/eval_seen_unseen_ablation_ordered_no_gs_no_gtav/g' \
    -e 's/_seen_no_gs_no_gtav/_seen_ordered_no_gs_no_gtav/g' \
    -e 's/_unseen_no_gs_no_gtav/_unseen_ordered_no_gs_no_gtav/g' \
    "$dst"

  chmod +x "$dst"
  echo "[OK] $dst"
done

cat > STEER-VLN/run_steer_vln_final_eval_seen_unseen_all_ordered_no_gs_no_gtav.sh <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${OPENFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

python STEER-VLN/make_openfly_ordered_no_gs_no_gtav_eval_json.py \
  --seen dataset/Annotation/seen.json \
  --unseen dataset/Annotation/unseen.json \
  --out_dir dataset/Annotation/filtered

echo "============================================================"
echo "[RUN ALL ORDERED NO-GS-NO-GTAV]"
echo "Order inside each split: env_airsim_26 -> other AirSim -> UE"
echo "============================================================"

bash STEER-VLN/run_steer_vln_final_eval_b0_seen_ordered_no_gs_no_gtav.sh
bash STEER-VLN/run_steer_vln_final_eval_b0_unseen_ordered_no_gs_no_gtav.sh

bash STEER-VLN/run_steer_vln_final_eval_e1_seen_ordered_no_gs_no_gtav.sh
bash STEER-VLN/run_steer_vln_final_eval_e1_unseen_ordered_no_gs_no_gtav.sh

bash STEER-VLN/run_steer_vln_final_eval_e2_seen_ordered_no_gs_no_gtav.sh
bash STEER-VLN/run_steer_vln_final_eval_e2_unseen_ordered_no_gs_no_gtav.sh

bash STEER-VLN/run_steer_vln_final_eval_e3_seen_ordered_no_gs_no_gtav.sh
bash STEER-VLN/run_steer_vln_final_eval_e3_unseen_ordered_no_gs_no_gtav.sh

echo "============================================================"
echo "[DONE] all ordered no-GS-no-GTAV evals finished"
echo "============================================================"
EOS

chmod +x STEER-VLN/run_steer_vln_final_eval_seen_unseen_all_ordered_no_gs_no_gtav.sh

echo "============================================================"
echo "[3] Check generated json env order"
echo "============================================================"
python - <<'PY'
import json
from pathlib import Path
from collections import Counter

def env(x):
    return str(x.get("image_path","")).split("/")[0]

for p in [
    Path("dataset/Annotation/filtered/seen_ordered_no_gs_no_gtav.json"),
    Path("dataset/Annotation/filtered/unseen_ordered_no_gs_no_gtav.json"),
]:
    data = json.loads(p.read_text())
    order = []
    for x in data:
        e = env(x)
        if e not in order:
            order.append(e)
    print("\n====", p, "====")
    print("total:", len(data))
    print("order:", " -> ".join(order) if order else "<empty>")
    for k, v in Counter(env(x) for x in data).items():
        print(f"{k}: {v}")
PY

echo "============================================================"
echo "[DONE]"
echo "Run examples:"
echo "  bash STEER-VLN/run_steer_vln_final_eval_b0_seen_ordered_no_gs_no_gtav.sh"
echo "  bash STEER-VLN/run_steer_vln_final_eval_seen_unseen_all_ordered_no_gs_no_gtav.sh"
echo "============================================================"
