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
)

from keyframe.openfly_feature_dual_dataset import (
    OpenFlyFeatureDualDataset,
    collate_openfly_feature_dual_batch,
)

from openfly_feature_trend_head import (
    StopPrestopLabelBuilder,
    load_trend_checkpoint,
    build_progress_from_batch,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="models/openfly-agent-7b")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--annotation_path", type=str, required=True)
    p.add_argument("--parquet_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--d_scale", type=float, default=12.0)
    p.add_argument("--z_scale", type=float, default=5.0)
    p.add_argument("--max_history", type=int, default=8)
    p.add_argument("--keyframe_mode", type=str, default="label", choices=["label", "residual", "learned_cache"])
    p.add_argument("--learned_keyframe_cache", type=str, default="")
    p.add_argument("--exclude_previous", action="store_true")

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cache_size", type=int, default=64)

    p.add_argument("--prestop_window", type=int, default=3)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--max_pred_rate", type=float, default=0.35)
    p.add_argument("--min_recall", type=float, default=0.10)

    return p.parse_args()


def binary_metrics(pred, target):
    pred = pred.long()
    target = target.long()

    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()

    total = max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "acc": float((tp + tn) / total),
        "pred_rate": float((tp + fp) / total),
        "true_rate": float((tp + fn) / total),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "count": int(total),
    }


def summarize_tensor(x):
    return {
        "min": float(x.min().item()),
        "p01": float(torch.quantile(x.float(), 0.01).item()),
        "p05": float(torch.quantile(x.float(), 0.05).item()),
        "p25": float(torch.quantile(x.float(), 0.25).item()),
        "mean": float(x.float().mean().item()),
        "p50": float(torch.quantile(x.float(), 0.50).item()),
        "p75": float(torch.quantile(x.float(), 0.75).item()),
        "p95": float(torch.quantile(x.float(), 0.95).item()),
        "p99": float(torch.quantile(x.float(), 0.99).item()),
        "max": float(x.max().item()),
    }


def select_rule(rows, max_pred_rate, min_recall):
    feasible = [
        r for r in rows
        if r["pred_rate"] <= max_pred_rate and r["recall"] >= min_recall
    ]

    if feasible:
        return max(feasible, key=lambda x: (x["f1"], x["recall"], -x["pred_rate"])), feasible

    # 如果没有满足约束的，选择 F1 最大的，避免返回全 0 的保守规则
    return max(rows, key=lambda x: (x["f1"], x["recall"], -x["pred_rate"])), feasible


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    processor, policy = load_openfly_local(args.model_path, device, dtype)
    head, cfg, ckpt = load_trend_checkpoint(args.checkpoint, device)
    head.eval()

    dataset = OpenFlyFeatureDualDataset(
        annotation_path=args.annotation_path,
        parquet_root=args.parquet_root,
        tokenizer=None,
        image_size=224,
        max_history=args.max_history,
        horizon=args.horizon,
        stride=1,
        max_episodes=None,
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

    label_builder = StopPrestopLabelBuilder(
        args.annotation_path,
        args.prestop_window,
        horizon=args.horizon,
    )

    stop_probs = []
    prestop_probs = []
    stop_targets = []
    prestop_targets = []
    action_margins = []
    wp_distances = []
    wp_yaws = []
    wp_zs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="collect"):
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

            logits = out["action_logits"]
            stop_action_logit = logits[:, 0]
            non_stop_max = logits[:, 1:].max(dim=1).values
            margin = stop_action_logit - non_stop_max

            waypoint = out["waypoint_pred"]
            d_raw = waypoint[:, 0].abs() * args.d_scale
            yaw_abs = waypoint[:, 1].abs()
            z_raw = waypoint[:, 2].abs() * args.z_scale

            stop_probs.append(torch.sigmoid(out["stop_logit"]).cpu())
            prestop_probs.append(torch.sigmoid(out["prestop_logit"]).cpu())
            stop_targets.append(stop_label.cpu())
            prestop_targets.append(prestop_label.cpu())
            action_margins.append(margin.cpu())
            wp_distances.append(d_raw.cpu())
            wp_yaws.append(yaw_abs.cpu())
            wp_zs.append(z_raw.cpu())

    stop_probs = torch.cat(stop_probs)
    prestop_probs = torch.cat(prestop_probs)
    stop_targets = torch.cat(stop_targets)
    prestop_targets = torch.cat(prestop_targets)
    action_margins = torch.cat(action_margins)
    wp_distances = torch.cat(wp_distances)
    wp_yaws = torch.cat(wp_yaws)
    wp_zs = torch.cat(wp_zs)

    stop_thresholds = [round(x / 100, 2) for x in range(20, 81, 5)]
    prestop_thresholds = [round(x / 100, 2) for x in range(20, 81, 5)]

    # very negative margin = almost disabled
    margin_thresholds = [-20.0, -10.0, -5.0, -3.0, -2.0, -1.0, 0.0, 0.5]

    # 999 = disabled distance gate
    distance_thresholds = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 999.0]

    rows = []

    for mode in ["prob_only", "prob_distance", "prob_margin", "prob_margin_distance"]:
        for ts in stop_thresholds:
            for tp in prestop_thresholds:
                for tm in margin_thresholds:
                    for td in distance_thresholds:
                        if mode == "prob_only":
                            pred = (stop_probs >= ts) & (prestop_probs >= tp)
                            used_tm = None
                            used_td = None
                        elif mode == "prob_distance":
                            pred = (
                                (stop_probs >= ts)
                                & (prestop_probs >= tp)
                                & (wp_distances <= td)
                            )
                            used_tm = None
                            used_td = td
                        elif mode == "prob_margin":
                            pred = (
                                (stop_probs >= ts)
                                & (prestop_probs >= tp)
                                & (action_margins >= tm)
                            )
                            used_tm = tm
                            used_td = None
                        else:
                            pred = (
                                (stop_probs >= ts)
                                & (prestop_probs >= tp)
                                & (action_margins >= tm)
                                & (wp_distances <= td)
                            )
                            used_tm = tm
                            used_td = td

                        m = binary_metrics(pred, stop_targets)
                        m.update({
                            "mode": mode,
                            "stop_threshold": ts,
                            "prestop_threshold": tp,
                            "margin_threshold": used_tm,
                            "distance_threshold": used_td,
                        })
                        rows.append(m)

                # 避免 prob_only / prob_distance 重复很多次
                if mode in ["prob_only", "prob_distance"]:
                    break

    selected, feasible = select_rule(
        rows,
        max_pred_rate=args.max_pred_rate,
        min_recall=args.min_recall,
    )

    by_mode = {}
    for mode in sorted(set(r["mode"] for r in rows)):
        mode_rows = [r for r in rows if r["mode"] == mode]
        by_mode[mode] = {
            "best_by_f1": max(mode_rows, key=lambda x: (x["f1"], x["recall"], -x["pred_rate"])),
            "top_by_f1": sorted(mode_rows, key=lambda x: x["f1"], reverse=True)[:10],
        }

    result = {
        "selected": selected,
        "constraints": {
            "max_pred_rate": args.max_pred_rate,
            "min_recall": args.min_recall,
            "num_feasible": len(feasible),
        },
        "top_by_f1": sorted(rows, key=lambda x: x["f1"], reverse=True)[:30],
        "top_feasible": sorted(feasible, key=lambda x: x["f1"], reverse=True)[:30],
        "by_mode": by_mode,
        "stats": {
            "stop_prob": summarize_tensor(stop_probs),
            "prestop_prob": summarize_tensor(prestop_probs),
            "action_margin": summarize_tensor(action_margins),
            "wp_distance": summarize_tensor(wp_distances),
            "wp_yaw_abs": summarize_tensor(wp_yaws),
            "wp_z_abs": summarize_tensor(wp_zs),
        },
    }

    with open(out_dir / "joint_sweep.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("===== selected =====")
    print(json.dumps(result["selected"], ensure_ascii=False, indent=2))
    print("===== by_mode best =====")
    print(json.dumps(
        {k: v["best_by_f1"] for k, v in by_mode.items()},
        ensure_ascii=False,
        indent=2,
    ))
    print("===== stats =====")
    print(json.dumps(result["stats"], ensure_ascii=False, indent=2))
    print(f"[OK] saved to {out_dir / 'joint_sweep.json'}")


if __name__ == "__main__":
    main()
