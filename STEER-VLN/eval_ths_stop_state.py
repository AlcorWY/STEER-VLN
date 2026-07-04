# ===== STEER_VLN force isolated import path =====
import sys as _steer_vln_sys
from pathlib import Path as _STEER_VLNPath

_STEER_VLN_FILE = _STEER_VLNPath(__file__).resolve()
_STEER_VLN_DIR = _STEER_VLN_FILE.parent
_STEER_VLN_ROOT = _STEER_VLN_DIR.parents[0]
_STEER_VLN_CODE = _STEER_VLN_ROOT / "code"
_STEER_VLN_TRAIN = _STEER_VLN_ROOT / "train"

for _p in [str(_STEER_VLN_DIR), str(_STEER_VLN_CODE), str(_STEER_VLN_ROOT), str(_STEER_VLN_TRAIN)]:
    if _p in _steer_vln_sys.path:
        _steer_vln_sys.path.remove(_p)

# priority:
#   STEER-VLN -> code -> project root -> train
_steer_vln_sys.path.insert(0, str(_STEER_VLN_ROOT))
_steer_vln_sys.path.insert(0, str(_STEER_VLN_CODE))
_steer_vln_sys.path.insert(0, str(_STEER_VLN_DIR))
_steer_vln_sys.path.append(str(_STEER_VLN_TRAIN))
# ===== end STEER_VLN force isolated import path =====

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from openfly_common import (
    set_seed,
    load_openfly_local,
    build_dataloader,
    move_labels_to_device,
    extract_openfly_features,
    aggregate_metrics,
)

from keyframe.openfly_feature_dual_dataset import (
    OpenFlyFeatureDualDataset,
    collate_openfly_feature_dual_batch,
)
from keyframe.openfly_feature_dual_head import action_metrics
from keyframe.waypoint_probe_model import waypoint_metrics
from lora_utils import load_lora_adapter_for_eval

from openfly_feature_trend_head import (
    StopPrestopLabelBuilder,
    binary_metrics_from_logits,
    aggregate_metric_dicts,
    load_trend_checkpoint,
    build_progress_from_batch,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_path", type=str, default="models/openfly-agent-7b")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--lora_adapter_path", type=str, default="")
    p.add_argument("--lora_target_path", type=str, default="")
    p.add_argument("--annotation_path", type=str, required=True)
    p.add_argument("--parquet_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--d_scale", type=float, default=12.0)
    p.add_argument("--z_scale", type=float, default=5.0)
    p.add_argument("--max_history", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)

    p.add_argument("--keyframe_mode", type=str, default="label", choices=["label", "residual", "learned_cache"])
    p.add_argument("--learned_keyframe_cache", type=str, default="")
    p.add_argument("--exclude_previous", action="store_true")

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cache_size", type=int, default=64)

    p.add_argument("--prestop_window", type=int, default=3)
    p.add_argument("--stop_threshold", type=float, default=0.5)
    p.add_argument("--prestop_threshold", type=float, default=0.5)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=7)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.fp16 and device.type == "cuda":
        dtype = torch.float16
    elif args.bf16 and device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print(f"[Device] {device}")
    print(f"[DType] {dtype}")
    print(f"[Checkpoint] {args.checkpoint}")

    processor, policy = load_openfly_local(args.model_path, device, dtype)

    if args.lora_adapter_path:
        policy = load_lora_adapter_for_eval(
            model=policy,
            adapter_dir=args.lora_adapter_path,
            lora_target_path=args.lora_target_path if args.lora_target_path else None,
        )
        policy.eval()
        print(f"[LoRA Eval] loaded adapter: {args.lora_adapter_path}", flush=True)

    head, cfg, ckpt = load_trend_checkpoint(args.checkpoint, device)
    head.eval()

    dataset = OpenFlyFeatureDualDataset(
        annotation_path=args.annotation_path,
        parquet_root=args.parquet_root,
        tokenizer=None,
        image_size=224,
        max_history=args.max_history,
        horizon=args.horizon,
        stride=args.stride,
        max_episodes=None if args.max_episodes <= 0 else args.max_episodes,
        keyframe_mode=args.keyframe_mode,
        exclude_previous=args.exclude_previous,
        d_scale=args.d_scale,
        z_scale=args.z_scale,
        cache_size=args.cache_size,
        learned_keyframe_cache=args.learned_keyframe_cache,
        prestop_window=getattr(args, "prestop_window", 3),
    )

    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_openfly_feature_dual_batch,
        prefetch_factor=2,
    )

    label_builder = StopPrestopLabelBuilder(args.annotation_path, args.prestop_window, horizon=args.horizon)

    action_metric_list = []
    waypoint_metric_list = []
    stop_metric_list = []
    prestop_metric_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            batch = move_labels_to_device(batch, device)
            stop_label, prestop_label = label_builder.labels_from_batch(batch, device)

            feat = extract_openfly_features(
                policy=policy,
                processor=processor,
                instructions=batch["instruction"],
                images_batch=batch["images"],
                device=device,
                dtype=dtype,
            )

            progress = build_progress_from_batch(batch, device)
            out = head(feat, progress=progress)

            action_metric_list.append(
                action_metrics(out["action_logits"], batch["action_label"])
            )

            waypoint_metric_list.append(
                waypoint_metrics(
                    pred=out["waypoint_pred"],
                    target_norm=batch["waypoint_label"],
                    target_raw=batch["waypoint_raw"],
                    mask=batch["waypoint_mask"],
                    d_scale=args.d_scale,
                    z_scale=args.z_scale,
                )
            )

            stop_metric_list.append(
                binary_metrics_from_logits(
                    out["stop_logit"],
                    stop_label,
                    threshold=args.stop_threshold,
                    prefix="stop",
                )
            )

            prestop_metric_list.append(
                binary_metrics_from_logits(
                    out["prestop_logit"],
                    prestop_label,
                    threshold=args.prestop_threshold,
                    prefix="prestop",
                )
            )

    metrics = {
        **aggregate_metrics(action_metric_list),
        **aggregate_metrics(waypoint_metric_list),
        **aggregate_metric_dicts(stop_metric_list),
        **aggregate_metric_dicts(prestop_metric_list),
    }

    metrics["checkpoint"] = args.checkpoint
    metrics["lora_adapter_path"] = args.lora_adapter_path
    metrics["keyframe_mode"] = args.keyframe_mode
    metrics["learned_keyframe_cache"] = args.learned_keyframe_cache

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
