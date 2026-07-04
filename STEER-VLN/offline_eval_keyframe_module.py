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
import csv
import json
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from keyframe.simple_tokenizer import SimpleTextTokenizer
from keyframe.keyframe_scorer import (
    AttentionKeyframeScorer,
    KeyframeScorerConfig,
    keyframe_soft_ce_loss,
    keyframe_top1_accuracy,
)
from train_integrated_full import IntegratedOpenFlyDataset, collate_integrated


def parse_args():
    p = argparse.ArgumentParser("Offline evaluation for STEER_VLN keyframe scorer module")
    p.add_argument("--final_model_dir", type=str, default="runs/STEER-VLN/train/final_model")
    p.add_argument("--annotation_path", type=str, default="dataset/Annotation/eval_airsim16_balanced_300.json")
    p.add_argument("--parquet_root", type=str, default="dataset/hf_openfly_airsim16/traj")
    p.add_argument("--output_dir", type=str, default="runs/STEER-VLN/offline_module_eval/keyframe")
    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_history", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min_timestep", type=int, default=2)
    p.add_argument("--cache_size", type=int, default=64)
    p.add_argument("--label_temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


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
        d_scale=12.0,
        z_scale=5.0,
        exclude_previous=True,
        prestop_window=3,
        label_temperature=args.label_temperature,
        cache_size=args.cache_size,
    )


def load_kfm(final_model_dir, device):
    final = Path(final_model_dir)
    ckpt_path = final / "keyframe_scorer_best.pt"
    vocab_path = final / "simple_tokenizer_vocab.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing keyframe checkpoint: {ckpt_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"missing keyframe vocab: {vocab_path}")

    tokenizer = SimpleTextTokenizer.load(vocab_path)
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

    model.to(device).eval()
    return tokenizer, model, ckpt_path, vocab_path


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model, ckpt_path, vocab_path = load_kfm(args.final_model_dir, device)

    ds_args = make_dataset_args(args)
    dataset = IntegratedOpenFlyDataset(ds_args, tokenizer)

    if len(dataset) <= 0:
        raise RuntimeError("No samples found for keyframe offline evaluation.")

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

    losses = []
    accs = []
    top1_probs = []
    entropies = []
    valid_counts = []
    rows = []

    for batch in tqdm(loader, desc="offline kfm eval"):
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        logits = model(
            history_images=batch["history_images"],
            current_image=batch["current_image"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            motion_feats=batch["motion_feats"],
            history_mask=batch["history_mask"],
        )

        loss = keyframe_soft_ce_loss(logits, batch["soft_labels"], batch["history_mask"])
        acc = keyframe_top1_accuracy(logits, batch["soft_labels"], batch["history_mask"])

        masked = logits.masked_fill(~batch["history_mask"].bool(), -1e4)
        probs = torch.softmax(masked.float(), dim=-1)
        pred = probs.argmax(dim=-1)
        target = batch["soft_labels"].argmax(dim=-1)

        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        vcnt = batch["history_mask"].float().sum(dim=-1)

        losses.append(float(loss.item()))
        accs.append(float(acc))
        top1_probs.extend(probs.max(dim=-1).values.detach().cpu().tolist())
        entropies.extend(entropy.detach().cpu().tolist())
        valid_counts.extend(vcnt.detach().cpu().tolist())

        for i, meta in enumerate(batch["meta"]):
            rows.append({
                "image_path": meta.get("image_path", ""),
                "parquet_path": meta.get("parquet_path", ""),
                "current_idx": int(meta.get("current_idx", -1)),
                "pred_local": int(pred[i].item()),
                "target_local": int(target[i].item()),
                "top1_prob": float(probs[i].max().item()),
                "entropy": float(entropy[i].item()),
                "valid_count": int(vcnt[i].item()),
                "correct": int(pred[i].item() == target[i].item()),
            })

    summary = {
        "module": "keyframe_scorer",
        "checkpoint": str(ckpt_path),
        "vocab": str(vocab_path),
        "annotation_path": args.annotation_path,
        "parquet_root": args.parquet_root,
        "num_samples": len(rows),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "top1_acc": float(np.mean(accs)) if accs else 0.0,
        "mean_top1_prob": float(np.mean(top1_probs)) if top1_probs else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_valid_history": float(np.mean(valid_counts)) if valid_counts else 0.0,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        with open(out_dir / "per_sample.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("=" * 80)
    print("[STEER-VLN OFFLINE KEYFRAME EVAL]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[SAVED] {out_dir / 'summary.json'}")
    print(f"[SAVED] {out_dir / 'per_sample.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
