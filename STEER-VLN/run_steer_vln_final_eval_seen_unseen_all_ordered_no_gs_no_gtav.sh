#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${OPENFLY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

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
