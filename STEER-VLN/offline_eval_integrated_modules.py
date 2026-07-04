#!/usr/bin/env python3
# ===== STEER_VLN isolated import path =====
import sys
from pathlib import Path

STEER_VLN_FILE = Path(__file__).resolve()
STEER_VLN_DIR = STEER_VLN_FILE.parent
ROOT = STEER_VLN_DIR.parents[0]
CODE = ROOT / "code"
TRAIN = ROOT / "train"

for p in [str(STEER_VLN_DIR), str(CODE), str(ROOT), str(TRAIN)]:
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(STEER_VLN_DIR))
sys.path.append(str(TRAIN))
# ===== end import path =====

import argparse
import csv
import json
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from extern.hf.configuration_prismatic import OpenFlyConfig
from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from keyframe.simple_tokenizer import SimpleTextTokenizer
from keyframe.keyframe_scorer import (
    AttentionKeyframeScorer,
    KeyframeScorerConfig,
    keyframe_soft_ce_loss,
    keyframe_top1_accuracy,
)
from keyframe.openfly_feature_dual_head import action_metrics
from keyframe.waypoint_probe_model import waypoint_metrics
from openfly_feature_trend_head import (
    load_trend_checkpoint,
    trend_head_loss,
    binary_metrics_from_logits,
    aggregate_metric_dicts,
    build_progress_from_batch,
)
from lora_utils import load_lora_adapter_for_eval
from train_integrated_full import (
    IntegratedOpenFlyDataset,
    collate_integrated,
    build_openfly_images_from_kfm,
    extract_openfly_features_eval,
    move_tensor_batch,
    move_processor_inputs,
)


ACTION_NAMES = {
    0: "stop",
    1: "forward_3m",
    2: "turn_left",
    3: "turn_right",
    4: "go_up",
    5: "go_down",
    6: "move_left",
    7: "move_right",
    8: "forward_6m",
    9: "forward_9m",
}


def parse_args():
    p = argparse.ArgumentParser("Offline evaluation for STEER_VLN integrated modules")
    p.add_argument("--final_model_dir", type=str, default="runs/STEER-VLN/train/final_model")
    p.add_argument("--model_path", type=str, default="models/openfly-agent-7b")
    p.add_argument("--annotation_path", type=str, default="dataset/Annotation/eval_airsim16_balanced_300.json")
    p.add_argument("--parquet_root", type=str, default="dataset/hf_openfly_airsim16/traj")
    p.add_argument("--output_dir", type=str, default="runs/STEER-VLN/offline_module_eval/integrated")
    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_history", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min_timestep", type=int, default=2)
    p.add_argument("--cache_size", type=int, default=64)
    p.add_argument("--label_temperature", type=float, default=0.7)
    p.add_argument("--topk", type=int, default=3)

    p.add_argument("--d_scale", type=float, default=12.0)
    p.add_argument("--z_scale", type=float, default=5.0)

    p.add_argument("--policy_modes", type=str, default="",
                   help="Optional comma-separated OpenFly predict_action offline modes: no_trend,oracle_trend. Empty disables final policy action eval.")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def choose_dtype(args, device):
    if device.type != "cuda":
        return torch.float32
    if args.fp16:
        return torch.float16
    if args.bf16:
        return torch.bfloat16
    return torch.bfloat16


def register_openfly_hf_classes():
    AutoConfig.register("openvla", OpenFlyConfig)
    AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)


def make_dataset_args(args):
    return SimpleNamespace(
        annotation_path=args.annotation_path,
        parquet_root=args.parquet_root,
        max_episodes=args.max_episodes,
        max_history=args.max_history,
        image_size=args.image_size,
        stride=args.stride,
        min_timestep=args.min_timestep,
        horizon=args.horizon,
        d_scale=args.d_scale,
        z_scale=args.z_scale,
        exclude_previous=True,
        prestop_window=3,
        label_temperature=args.label_temperature,
        cache_size=args.cache_size,
    )


def aggregate_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    return aggregate_metric_dicts(items)


def load_kfm(final_model_dir, tokenizer, device):
    ckpt_path = Path(final_model_dir) / "keyframe_scorer_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = KeyframeScorerConfig(**dict(ckpt["cfg"]))
    model = AttentionKeyframeScorer(cfg, padding_idx=tokenizer.pad_token_id)
    state = ckpt.get("model", ckpt.get("keyframe_scorer", None))
    if state is None:
        raise KeyError(f"keyframe checkpoint does not contain model/keyframe_scorer state: {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[KFM][WARN] missing keys: {missing[:8]} total={len(missing)}", flush=True)
    if unexpected:
        print(f"[KFM][WARN] unexpected keys: {unexpected[:8]} total={len(unexpected)}", flush=True)
    return model.to(device).eval(), ckpt_path


def load_policy_and_processor(args, device, dtype):
    register_openfly_hf_classes()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    policy = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)

    adapter = Path(args.final_model_dir) / "lora_adapter_best"
    if adapter.exists():
        policy = load_lora_adapter_for_eval(policy, str(adapter))

    policy.eval()
    return processor, policy


def convert_to_action_id(action) -> int:
    action = np.asarray(action).round().astype(int)
    action_dict = {
        0: np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=int),
        1: np.array([0, 3, 0, 0, 0, 0, 0, 0], dtype=int),
        2: np.array([0, 0, 15, 0, 0, 0, 0, 0], dtype=int),
        3: np.array([0, 0, 0, 15, 0, 0, 0, 0], dtype=int),
        4: np.array([0, 0, 0, 0, 2, 0, 0, 0], dtype=int),
        5: np.array([0, 0, 0, 0, 0, 2, 0, 0], dtype=int),
        6: np.array([0, 0, 0, 0, 0, 0, 5, 0], dtype=int),
        7: np.array([0, 0, 0, 0, 0, 0, 0, 5], dtype=int),
        8: np.array([0, 6, 0, 0, 0, 0, 0, 0], dtype=int),
        9: np.array([0, 9, 0, 0, 0, 0, 0, 0], dtype=int),
    }
    for idx, value in action_dict.items():
        if np.array_equal(action, value):
            return int(idx)
    return 0


def _predict_policy_actions(policy, processor, texts, images_batch, device, dtype):
    preds = []
    for text, images in zip(texts, images_batch):
        inputs = processor(str(text).lower(), images, return_tensors="pt")
        inputs = move_processor_inputs(inputs, device=device, dtype=dtype)
        action = policy.predict_action(**inputs, unnorm_key="vlnv1", do_sample=False)
        preds.append(convert_to_action_id(action))
    return preds


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final = Path(args.final_model_dir)
    for required in ["keyframe_scorer_best.pt", "simple_tokenizer_vocab.json", "m3c_trend_head_best.pt"]:
        if not (final / required).exists():
            raise FileNotFoundError(f"missing {final / required}")
    if not (final / "lora_adapter_best").is_dir():
        raise FileNotFoundError(f"missing {final / 'lora_adapter_best'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(args, device)

    tokenizer = SimpleTextTokenizer.load(final / "simple_tokenizer_vocab.json")
    dataset = IntegratedOpenFlyDataset(make_dataset_args(args), tokenizer)

    if len(dataset) <= 0:
        raise RuntimeError("No samples found for integrated offline evaluation.")

    if args.max_samples and args.max_samples > 0 and len(dataset) > args.max_samples:
        dataset = Subset(dataset, list(range(args.max_samples)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda b: collate_integrated(b, pad_token_id=tokenizer.pad_token_id),
    )

    kfm_model, kfm_path = load_kfm(final, tokenizer, device)
    trend_head, trend_cfg, trend_ckpt = load_trend_checkpoint(final / "m3c_trend_head_best.pt", device)
    trend_head.eval()

    processor, policy = load_policy_and_processor(args, device, dtype)

    kfm_losses, kfm_accs = [], []
    action_metric_list, waypoint_metric_list = [], []
    stop_metric_list, prestop_metric_list = [], []

    policy_modes = [x.strip() for x in args.policy_modes.split(",") if x.strip()]
    policy_stats = {m: {"correct": 0, "total": 0} for m in policy_modes}

    rows = []

    for batch in tqdm(loader, desc="offline integrated module eval"):
        batch = move_tensor_batch(batch, device)

        kfm_logits = kfm_model(
            history_images=batch["history_images"],
            current_image=batch["current_image"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            motion_feats=batch["motion_feats"],
            history_mask=batch["history_mask"],
        )

        kfm_loss = keyframe_soft_ce_loss(kfm_logits, batch["soft_labels"], batch["history_mask"])
        kfm_acc = keyframe_top1_accuracy(kfm_logits, batch["soft_labels"], batch["history_mask"])
        kfm_losses.append(float(kfm_loss.item()))
        kfm_accs.append(float(kfm_acc))

        images_batch = build_openfly_images_from_kfm(batch, kfm_logits, topk=args.topk)

        features = extract_openfly_features_eval(
            policy=policy,
            processor=processor,
            instructions=batch["instruction"],
            images_batch=images_batch,
            device=device,
            dtype=dtype,
        )

        progress = build_progress_from_batch(batch, device)
        out = trend_head(features, progress=progress)

        action_metric_list.append(action_metrics(out["action_logits"], batch["action_label"]))
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
        stop_metric_list.append(binary_metrics_from_logits(out["stop_logit"], batch["stop_label"], prefix="stop"))
        prestop_metric_list.append(binary_metrics_from_logits(out["prestop_logit"], batch["prestop_label"], prefix="prestop"))

        mode_preds = {}
        if "no_trend" in policy_modes:
            mode_preds["no_trend"] = _predict_policy_actions(
                policy, processor, batch["original_instruction"], images_batch, device, dtype
            )
        if "oracle_trend" in policy_modes:
            mode_preds["oracle_trend"] = _predict_policy_actions(
                policy, processor, batch["instruction"], images_batch, device, dtype
            )

        labels = batch["action_label"].detach().cpu().tolist()
        for mode, preds in mode_preds.items():
            for pred, lab in zip(preds, labels):
                policy_stats[mode]["correct"] += int(int(pred) == int(lab))
                policy_stats[mode]["total"] += 1

        trend_pred = out["action_logits"].argmax(dim=-1).detach().cpu().tolist()
        stop_prob = torch.sigmoid(out["stop_logit"]).detach().cpu().flatten().tolist()
        prestop_prob = torch.sigmoid(out["prestop_logit"]).detach().cpu().flatten().tolist()
        wp_pred = out["waypoint_pred"].detach().cpu().tolist()

        for i, meta in enumerate(batch["meta"]):
            row = {
                "image_path": meta.get("image_path", ""),
                "parquet_path": meta.get("parquet_path", ""),
                "current_idx": int(meta.get("current_idx", -1)),
                "action_label": int(labels[i]),
                "trend_aux_action_pred": int(trend_pred[i]),
                "trend_aux_action_correct": int(int(trend_pred[i]) == int(labels[i])),
                "stop_label": float(batch["stop_label"][i].detach().cpu().item()),
                "stop_prob": float(stop_prob[i]),
                "prestop_label": float(batch["prestop_label"][i].detach().cpu().item()),
                "prestop_prob": float(prestop_prob[i]),
                "wp_pred_d_norm": float(wp_pred[i][0]),
                "wp_pred_yaw_norm": float(wp_pred[i][1]),
                "wp_pred_z_norm": float(wp_pred[i][2]),
            }
            for mode, preds in mode_preds.items():
                row[f"policy_{mode}_pred"] = int(preds[i])
                row[f"policy_{mode}_correct"] = int(int(preds[i]) == int(labels[i]))
            rows.append(row)

    summary = {
        "module": "integrated_modules_offline",
        "note": (
            "Trend action/waypoint/stop metrics evaluate the auxiliary trend head. "
            "They are not the final online SR/OSR metrics. Optional policy_* metrics use OpenFly/LoRA predict_action offline on recorded frames."
        ),
        "final_model_dir": str(final),
        "model_path": args.model_path,
        "annotation_path": args.annotation_path,
        "parquet_root": args.parquet_root,
        "num_samples": len(rows),
        "kfm_loss": float(np.mean(kfm_losses)) if kfm_losses else 0.0,
        "kfm_top1_acc": float(np.mean(kfm_accs)) if kfm_accs else 0.0,
        **aggregate_metrics(action_metric_list),
        **aggregate_metrics(waypoint_metric_list),
        **aggregate_metric_dicts(stop_metric_list),
        **aggregate_metric_dicts(prestop_metric_list),
    }

    for mode, st in policy_stats.items():
        total = max(int(st["total"]), 1)
        summary[f"policy_{mode}_action_acc"] = float(st["correct"] / total)
        summary[f"policy_{mode}_correct"] = int(st["correct"])
        summary[f"policy_{mode}_total"] = int(st["total"])

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        keys = sorted(set().union(*[r.keys() for r in rows]))
        with open(out_dir / "per_sample.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    print("=" * 80)
    print("[STEER-VLN OFFLINE INTEGRATED MODULE EVAL]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[SAVED] {out_dir / 'summary.json'}")
    print(f"[SAVED] {out_dir / 'per_sample.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
