#!/usr/bin/env python3
"""Offline ablation/experiment runner for STEER_VLN modules.

This script invokes existing offline eval tools under `STEER-VLN/` (for example
`eval_ths_stop_state.py`) while setting module flags such as LoRA adapter,
keyframe mode, and trend head usage. It writes logs under `runs/STEER-VLN/logs/`.

Usage examples:

# Baseline OpenFly (no LoRA, no trend head)
python3 STEER-VLN/offline_ablation_runner.py --name openfly_baseline \
  --model_path models/openfly-agent-7b \
  --annotation_path dataset/Annotation/eval_airsim16_balanced_300.json \
  --parquet_root dataset/hf_openfly_airsim16/traj \
  --no_lora --no_trend

# LoRA adapter enabled
python3 STEER-VLN/offline_ablation_runner.py --name lora_test \
  --model_path models/openfly-agent-7b \
  --annotation_path dataset/Annotation/eval_airsim16_balanced_300.json \
  --parquet_root dataset/hf_openfly_airsim16/traj \
  --lora_adapter_path runs/STEER-VLN/train/lora_full/checkpoint 

"""
import argparse
import os
from pathlib import Path
import shlex

from steer_vln_runner import run_with_log, abs_path, ensure_dir

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = "runs/STEER-VLN/eval_ablation"


def build_eval_ths_cmd(args, out_dir: str):
    cmd = [
        "python3",
        "STEER-VLN/eval_ths_stop_state.py",
        "--model_path", args.model_path,
        "--checkpoint", args.checkpoint,
        "--annotation_path", args.annotation_path,
        "--parquet_root", args.parquet_root,
        "--output_dir", out_dir,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--horizon", str(args.horizon),
    ]

    if args.lora_adapter_path:
        cmd += ["--lora_adapter_path", args.lora_adapter_path]
    if args.lora_target_path:
        cmd += ["--lora_target_path", args.lora_target_path]

    if args.keyframe_mode:
        cmd += ["--keyframe_mode", args.keyframe_mode]
    if args.exclude_previous:
        cmd += ["--exclude_previous"]

    if args.fp16:
        cmd += ["--fp16"]
    if args.bf16:
        cmd += ["--bf16"]

    if args.max_episodes:
        cmd += ["--max_episodes", str(args.max_episodes)]

    return cmd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, required=True, help="short name for this experiment")
    p.add_argument("--model_path", type=str, default="models/openfly-agent-7b")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--lora_adapter_path", type=str, default="")
    p.add_argument("--lora_target_path", type=str, default="")
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--no_trend", action="store_true")
    p.add_argument("--keyframe_mode", type=str, default="label", choices=["label", "residual", "learned_cache"]) 
    p.add_argument("--exclude_previous", action="store_true")
    p.add_argument("--annotation_path", type=str, required=True)
    p.add_argument("--parquet_root", type=str, required=True)
    p.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    # Batch options: comma-separated lists. If provided, script will run all combinations.
    p.add_argument("--batch", action="store_true", help="Run a batch of ablation combinations")
    p.add_argument("--keyframe_modes", type=str, default="", help="comma-separated keyframe modes to sweep")
    p.add_argument("--lora_adapter_paths", type=str, default="", help="comma-separated LoRA adapter paths (empty means no LoRA)")
    p.add_argument("--trend_options", type=str, default="", help="comma-separated trend options: 1 or 0")
    p.add_argument("--exclude_prev_options", type=str, default="", help="comma-separated exclude_previous options: 1 or 0")
    p.add_argument("--checkpoints", type=str, default="", help="comma-separated checkpoint paths to evaluate")
    return p.parse_args()


def main():
    args = parse_args()

    # Prepare output directory
    out_root = Path(args.output_root) / args.name
    ensure_dir(str(out_root))

    # Prepare environment variables for module toggles
    # By default enable trend if a checkpoint is provided; allow --no_trend to disable
    if args.no_trend or not args.checkpoint:
        os.environ.setdefault("TREND_CONDITIONED", "0")
    else:
        os.environ.setdefault("TREND_CONDITIONED", "1")

    # LoRA adapter: if --no_lora, ensure unset; otherwise pass path if provided
    if args.no_lora:
        os.environ.pop("LORA_ADAPTER_PATH", None)
        os.environ.pop("LORA_TARGET_PATH", None)
    else:
        if args.lora_adapter_path:
            os.environ.setdefault("LORA_ADAPTER_PATH", args.lora_adapter_path)
        if args.lora_target_path:
            os.environ.setdefault("LORA_TARGET_PATH", args.lora_target_path)

    # Keyframe mode
    os.environ.setdefault("KEYFRAME_MODE", args.keyframe_mode)

    # Offline mode for HF
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Batch mode: generate combinations and run sequentially
    if args.batch:
        import itertools

        keyframe_modes = [k for k in args.keyframe_modes.split(",") if k] or [args.keyframe_mode]
        lora_paths = [p for p in args.lora_adapter_paths.split(",") if p] or ([""])
        trend_opts = [t for t in args.trend_options.split(",") if t] or ([os.environ.get("TREND_CONDITIONED", "0")])
        excl_opts = [e for e in args.exclude_prev_options.split(",") if e] or (["0", "1"])
        checkpoints = [c for c in args.checkpoints.split(",") if c] or ([args.checkpoint])

        combos = list(itertools.product(keyframe_modes, lora_paths, trend_opts, excl_opts, checkpoints))
        print(f"Running batch with {len(combos)} combos")

        for km, lora_p, trend_o, excl_o, ckpt in combos:
            combo_name = f"{args.name}_kf-{km.replace('/', '_')}_lora-{('none' if lora_p=='' else Path(lora_p).name)}_trend-{trend_o}_excl-{excl_o}_ckpt-{('none' if ckpt=='' else Path(ckpt).stem)}"
            combo_out = out_root / combo_name
            ensure_dir(str(combo_out))

            # set env / flags per combo
            if lora_p:
                os.environ["LORA_ADAPTER_PATH"] = lora_p
            else:
                os.environ.pop("LORA_ADAPTER_PATH", None)

            if ckpt:
                args_checkpoint = ckpt
            else:
                args_checkpoint = ""

            if trend_o == "1":
                os.environ["TREND_CONDITIONED"] = "1"
            else:
                os.environ["TREND_CONDITIONED"] = "0"

            if excl_o == "1":
                exclude_previous = True
            else:
                exclude_previous = False

            # build cmd with per-combo values
            cmd = [
                "python3",
                "STEER-VLN/eval_ths_stop_state.py",
                "--model_path", args.model_path,
                "--checkpoint", args_checkpoint,
                "--annotation_path", args.annotation_path,
                "--parquet_root", args.parquet_root,
                "--output_dir", str(combo_out),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--horizon", str(args.horizon),
                "--keyframe_mode", km,
            ]

            if lora_p:
                cmd += ["--lora_adapter_path", lora_p]
            if args.bf16:
                cmd += ["--bf16"]
            if args.fp16:
                cmd += ["--fp16"]
            if exclude_previous:
                cmd += ["--exclude_previous"]

            log_path = f"runs/STEER-VLN/logs/eval_ablation_{combo_name}.log"
            run_with_log(cmd, log_path)

        print("Batch finished")
        return

    # Build command to run the offline trend/stop evaluation
    out_dir = str(out_root / "eval_ths_stop_state")
    cmd = build_eval_ths_cmd(args, out_dir)

    log_path = f"runs/STEER-VLN/logs/eval_ablation_{args.name}.log"
    run_with_log(cmd, log_path)

    print(f"Experiment {args.name} finished. Results in: {out_dir}")


if __name__ == '__main__':
    main()
