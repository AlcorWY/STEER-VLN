#!/usr/bin/env python3
# ===== STEER_VLN isolated import path =====
import sys
from pathlib import Path

STEER_VLN_FILE = Path(__file__).resolve()
STEER_VLN_DIR = STEER_VLN_FILE.parent
ROOT = STEER_VLN_DIR.parents[0]
CODE = ROOT / "code"

for p in [str(STEER_VLN_DIR), str(CODE), str(ROOT)]:
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(STEER_VLN_DIR))
# ===== end import path =====

import argparse
import json
import re
from typing import Dict, List


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def has_all(text: str, items: List[str]) -> bool:
    return all(x in text for x in items)


def record(checks, name, ok, detail):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})


def main():
    ap = argparse.ArgumentParser("Static STEER-VLN scheme checker")
    ap.add_argument("--steer_vln_dir", type=str, default="STEER-VLN")
    ap.add_argument("--final_model_dir", type=str, default="runs/STEER-VLN/train/final_model")
    ap.add_argument("--output_dir", type=str, default="runs/STEER-VLN/offline_module_eval/check_scheme")
    args = ap.parse_args()

    steer_vln = Path(args.steer_vln_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    checks = []

    train_py = read(steer_vln / "train_integrated_full.py")
    lora_py = read(steer_vln / "lora_utils.py")
    train_sh = read(steer_vln / "run_steer_vln_final_train.sh")
    eval_sh = read(steer_vln / "run_steer_vln_final_eval_seen_unseen_all_ordered_no_gs_no_gtav.sh")
    eval_hd = read(steer_vln / "eval_hd_lora.py")
    trend_rt = read(steer_vln / "openfly_trend_state_runtime.py")

    record(checks, "steer_vln_dir_exists", steer_vln.exists(), steer_vln)
    record(checks, "split_train_eval_scripts", (steer_vln / "run_steer_vln_final_train.sh").exists() and (steer_vln / "run_steer_vln_final_eval_seen_unseen_all_ordered_no_gs_no_gtav.sh").exists(),
           "train/eval are separated")

    qkvo = "q_proj,k_proj,v_proj,o_proj"
    record(checks, "train_default_lora_qkvo", qkvo in train_py, "train_integrated_full.py default/args contain q,k,v,o")
    record(checks, "runner_passes_lora_qkvo", qkvo in train_sh, "run_steer_vln_final_train.sh passes q,k,v,o")
    record(checks, "lora_utils_qkvo_available", qkvo in lora_py, "lora_utils can attach q,k,v,o")

    record(checks, "version_a_integrated_loop",
           has_all(train_py, ["Keyframe Scorer", "OpenFly/LoRA", "Temporal Trend Head"]) or
           has_all(train_py, ["kfm_model", "trend_head", "policy"]),
           "KFM + LoRA policy + trend head exist in one trainer")

    record(checks, "kfm_joint_loss_present",
           "kfm_loss_weight" in train_py and "keyframe_soft_ce_loss" in train_py,
           "Keyframe scorer is optimized by soft keyframe loss inside integrated training")

    record(checks, "online_kfm_topk_fusion_present",
           "build_openfly_images_from_kfm" in train_py and "topk" in train_py,
           "Training builds online top-k/fused keyframe rather than using a prebuilt cache")

    record(checks, "no_required_keyframe_cache_in_integrated_train",
           "--learned_keyframe_cache" not in train_py and "learned_keyframe_cache" not in train_sh,
           "Integrated trainer does not require keyframe cache")

    record(checks, "trend_auxiliary_no_override_runtime",
           "never overrides" in trend_rt.lower() and "conditioned_text" in trend_rt,
           "Trend runtime returns auxiliary trend text/state and does not force final action")

    record(checks, "final_action_openfly_eval",
           "policy.predict_action" in eval_hd and "ACTION_MODE" in eval_hd,
           "Final eval can use OpenFly/LoRA predict_action as final action source")

    record(checks, "hook_hidden_feature_oom_fix",
           "find_language_final_norm" in train_py and "output_hidden_states=False" in train_py,
           "OpenFly feature extraction uses final-norm hook to avoid storing all hidden states")

    record(checks, "save_before_validation",
           "pre_val_safety_save" in train_py,
           "Trainer saves last checkpoint before validation")

    record(checks, "aggregate_metrics_defined",
           "def aggregate_metrics" in train_py,
           "Validation metric aggregation compatibility wrapper exists")

    record(checks, "steer_vln_paths_only",
           "runs/STEER-VLN" in train_sh and "STEER-VLN/" in train_sh and "STEER-VLN/" in eval_sh
           and ("/".join(["code", "STEER-VLN"]) not in train_sh)
           and ("/".join(["code", "STEER-VLN"]) not in eval_sh),
           "Train/eval scripts use root-level STEER-VLN source and run directories")

    old_refs = []
    old_terms = [
        "/".join(["code", "STEER-VLN"]),
        "/".join(["runs", "code", "STEER-VLN"]),
        "/".join(["code", "e" + "7"]),
        "/".join(["runs", "code", "e" + "7"]),
        "E" + "7_FINAL_MODEL_DIR",
    ]
    for p in steer_vln.rglob("*"):
        if p.name == "check_steer_vln_scheme.py":
            continue
        if p.is_file() and p.suffix in [".py", ".sh", ".md"]:
            txt = read(p)
            if any(term in txt for term in old_terms):
                old_refs.append(str(p.relative_to(steer_vln)))
    record(checks, "no_old_runtime_path_refs", len(old_refs) == 0, old_refs)

    final_dir = Path(args.final_model_dir)
    required_files = [
        "keyframe_scorer_best.pt",
        "simple_tokenizer_vocab.json",
        "m3c_trend_head_best.pt",
        "integrated_full_best.pt",
        "final_model_manifest.json",
    ]
    required_dirs = ["lora_adapter_best"]
    missing = []
    for f in required_files:
        if not (final_dir / f).exists():
            missing.append(str(final_dir / f))
    for d in required_dirs:
        if not (final_dir / d).is_dir():
            missing.append(str(final_dir / d))
    record(checks, "final_model_artifacts_available", len(missing) == 0,
           "missing=" + json.dumps(missing, ensure_ascii=False))

    ok_count = sum(1 for c in checks if c["ok"])
    fail_count = len(checks) - ok_count

    report = {
        "scheme": "STEER-VLN Version-A integrated training with LoRA q,k,v,o",
        "steer_vln_dir": str(steer_vln),
        "final_model_dir": str(final_dir),
        "ok": fail_count == 0,
        "ok_count": ok_count,
        "fail_count": fail_count,
        "checks": checks,
    }

    (out / "steer_vln_scheme_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# STEER-VLN Scheme Check",
        "",
        f"- Overall: {'PASS' if report['ok'] else 'WARN/FAIL'}",
        f"- Passed: {ok_count}",
        f"- Failed: {fail_count}",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for c in checks:
        lines.append(f"| {c['name']} | {'OK' if c['ok'] else 'FAIL'} | {str(c['detail']).replace('|','/')} |")
    (out / "steer_vln_scheme_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 80)
    print("[STEER-VLN SCHEME CHECK]")
    print(f"Overall: {'PASS' if report['ok'] else 'WARN/FAIL'}")
    print(f"Saved: {out / 'steer_vln_scheme_check.json'}")
    print(f"Saved: {out / 'steer_vln_scheme_check.md'}")
    print("=" * 80)
    for c in checks:
        print(f"[{'OK' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")

    if fail_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
