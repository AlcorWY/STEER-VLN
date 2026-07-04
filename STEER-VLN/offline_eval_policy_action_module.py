#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
STEER_VLN = ROOT / "STEER-VLN"
TRAIN = ROOT / "train"

for p in [str(STEER_VLN), str(ROOT / "code"), str(ROOT), str(TRAIN)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from extern.hf.configuration_prismatic import OpenFlyConfig
from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from train_integrated_full import IntegratedOpenFlyDataset
from keyframe.simple_tokenizer import SimpleTextTokenizer
from keyframe.keyframe_scorer import AttentionKeyframeScorer, KeyframeScorerConfig
from lora_utils import load_lora_adapter_for_eval


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


def register_openfly():
    AutoConfig.register("openvla", OpenFlyConfig)
    AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)


def load_tokenizer(vocab_path):
    vocab_path = Path(vocab_path)

    if hasattr(SimpleTextTokenizer, "load"):
        return SimpleTextTokenizer.load(vocab_path)
    if hasattr(SimpleTextTokenizer, "from_file"):
        return SimpleTextTokenizer.from_file(vocab_path)
    if hasattr(SimpleTextTokenizer, "from_vocab_file"):
        return SimpleTextTokenizer.from_vocab_file(vocab_path)

    data = json.load(open(vocab_path, "r", encoding="utf-8"))
    tok = SimpleTextTokenizer()

    if isinstance(data, dict) and "token_to_id" in data:
        tok.token_to_id = data["token_to_id"]
    elif isinstance(data, dict):
        tok.token_to_id = data
    else:
        raise RuntimeError(f"Cannot load vocab: {vocab_path}")

    tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
    tok.pad_token_id = tok.token_to_id.get("<pad>", 0)
    return tok


def make_dataset(args, tokenizer):
    ds_args = SimpleNamespace(
        annotation_path=args.annotation_path,
        parquet_root=args.parquet_root,
        max_episodes=0,
        max_history=args.max_history,
        image_size=args.image_size,
        stride=args.stride,
        min_timestep=args.min_timestep,
        horizon=args.horizon,
        d_scale=args.d_scale,
        z_scale=args.z_scale,
        exclude_previous=True,
        prestop_window=3,
        label_temperature=0.7,
        cache_size=32,
    )
    return IntegratedOpenFlyDataset(ds_args, tokenizer)


def tensor_to_pil_01(x):
    x = x.detach().float().cpu().clamp(0, 1)
    arr = x.permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def masked_softmax(logits, mask):
    logits = logits.masked_fill(~mask.bool(), -1e4)
    return torch.softmax(logits, dim=-1)


def load_kfm(final_model_dir, device):
    final_model_dir = Path(final_model_dir)
    ckpt = torch.load(final_model_dir / "keyframe_scorer_best.pt", map_location="cpu")
    cfg = KeyframeScorerConfig(**ckpt.get("cfg", {}))
    tokenizer = load_tokenizer(final_model_dir / "simple_tokenizer_vocab.json")

    model = AttentionKeyframeScorer(cfg, padding_idx=getattr(tokenizer, "pad_token_id", 0))
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer


def build_kfm_bundle(sample, kfm, device, topk=3):
    batch = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = kfm(
            history_images=batch["history_images"],
            current_image=batch["current_image"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            motion_feats=batch["motion_feats"],
            history_mask=batch["history_mask"],
        )
        probs = masked_softmax(logits, batch["history_mask"])[0].detach().cpu()

    valid = torch.nonzero(sample["history_mask"].bool(), as_tuple=False).flatten()
    topk = min(topk, int(valid.numel()))

    if topk > 0:
        vals, pos = torch.topk(probs[valid], k=topk)
        local = valid[pos].tolist()
        weights = vals / vals.sum().clamp_min(1e-8)

        fused = torch.zeros_like(sample["history_raw"][local[0]])
        for w, idx in zip(weights, local):
            fused = fused + float(w) * sample["history_raw"][idx]
        fused = fused.clamp(0, 1)
    else:
        local = []
        vals = torch.tensor([])
        weights = torch.tensor([])
        fused = sample["previous_raw"]

    bundle = [
        tensor_to_pil_01(fused),
        tensor_to_pil_01(sample["previous_raw"]),
        tensor_to_pil_01(sample["current_raw"]),
    ]

    return bundle, local, vals.tolist()


def build_residual_bundle(sample):
    valid = torch.nonzero(sample["history_mask"].bool(), as_tuple=False).flatten()
    if valid.numel() > 0:
        older = sample["history_raw"][int(valid[-1].item())]
    else:
        older = sample["previous_raw"]

    return [
        tensor_to_pil_01(older),
        tensor_to_pil_01(sample["previous_raw"]),
        tensor_to_pil_01(sample["current_raw"]),
    ]


def convert_to_action_id(action):
    action_dict = {
        0: np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32),
        1: np.array([0, 3, 0, 0, 0, 0, 0, 0]).astype(np.float32),
        2: np.array([0, 0, 15, 0, 0, 0, 0, 0]).astype(np.float32),
        3: np.array([0, 0, 0, 15, 0, 0, 0, 0]).astype(np.float32),
        4: np.array([0, 0, 0, 0, 2, 0, 0, 0]).astype(np.float32),
        5: np.array([0, 0, 0, 0, 0, 2, 0, 0]).astype(np.float32),
        6: np.array([0, 0, 0, 0, 0, 0, 5, 0]).astype(np.float32),
        7: np.array([0, 0, 0, 0, 0, 0, 0, 5]).astype(np.float32),
        8: np.array([0, 6, 0, 0, 0, 0, 0, 0]).astype(np.float32),
        9: np.array([0, 9, 0, 0, 0, 0, 0, 0]).astype(np.float32),
    }

    arr = np.asarray(action).round().astype(np.float32).reshape(-1)
    if arr.shape[0] > 8:
        arr = arr[:8]

    for idx, ref in action_dict.items():
        if np.array_equal(arr, ref):
            return idx

    return 0


def predict_policy_action(policy, processor, text, images, device, dtype):
    inputs = processor(str(text).lower(), images).to(device, dtype=dtype)
    with torch.no_grad():
        action = policy.predict_action(**inputs, unnorm_key="vlnv1", do_sample=False)
    return convert_to_action_id(action), np.asarray(action).tolist()


def load_policy_and_lora(args, device, dtype):
    register_openfly()

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

    policy = load_lora_adapter_for_eval(
        policy,
        str(Path(args.final_model_dir) / "lora_adapter_best"),
    )

    policy.eval()
    return processor, policy


def draw_label(draw, xy, text, fill=(255, 255, 255), bg=(20, 20, 20)):
    font = ImageFont.load_default()
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3], fill=bg)
    draw.text((x, y), text, fill=fill, font=font)


def save_policy_visual(images, title, info_lines, out_path):
    cell = 224
    margin = 18
    label_h = 28
    title_h = 52
    info_h = max(160, 22 * len(info_lines) + 30)

    W = 3 * cell + 4 * margin
    H = title_h + cell + label_h + margin + info_h

    canvas = Image.new("RGB", (W, H), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, W, title_h], fill=(30, 30, 30))
    draw.text((margin, 18), title, fill=(255, 255, 255), font=ImageFont.load_default())

    labels = ["keyframe", "previous", "current"]

    y = title_h + margin
    for i, img in enumerate(images):
        x = margin + i * (cell + margin)
        canvas.paste(img.resize((cell, cell)), (x, y))
        draw.rectangle([x, y, x + cell, y + cell], outline=(40, 80, 220), width=4)
        draw_label(draw, (x, y + cell + 6), labels[i], bg=(40, 80, 220))

    y_info = y + cell + label_h + margin
    for line in info_lines:
        draw.text((margin, y_info), str(line), fill=(0, 0, 0), font=ImageFont.load_default())
        y_info += 22

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_model_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--annotation_path", required=True)
    ap.add_argument("--parquet_root", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--modes", default="residual_no_trend,learned_no_trend,learned_oracle_trend")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--max_visual_cases", type=int, default=24)
    ap.add_argument("--topk", type=int, default=3)

    ap.add_argument("--max_history", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min_timestep", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--d_scale", type=float, default=12.0)
    ap.add_argument("--z_scale", type=float, default=5.0)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--attn_implementation", default="flash_attention_2")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and args.bf16:
        dtype = torch.bfloat16
    elif device.type == "cuda" and args.fp16:
        dtype = torch.float16
    elif device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print("[POLICY OFFLINE] device:", device)
    print("[POLICY OFFLINE] dtype:", dtype)

    kfm, tokenizer = load_kfm(args.final_model_dir, device)
    dataset = make_dataset(args, tokenizer)

    processor, policy = load_policy_and_lora(args, device, dtype)

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    n = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))

    per_rows = []
    summary_by_mode = {}

    vis_root = out / "visualizations"
    vis_root.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        correct = 0
        total = 0
        pred_counter = Counter()
        gt_counter = Counter()

        mode_vis = vis_root / mode
        mode_vis.mkdir(parents=True, exist_ok=True)

        print(f"[POLICY OFFLINE] mode={mode}, samples={n}", flush=True)

        for idx in range(n):
            sample = dataset[idx]

            gt = int(sample["action_label"].item())
            gt_name = ACTION_NAMES.get(gt, str(gt))

            if mode == "residual_no_trend":
                images = build_residual_bundle(sample)
                text = sample["original_instruction"]
                topk_local = []
                topk_prob = []
            elif mode == "learned_no_trend":
                images, topk_local, topk_prob = build_kfm_bundle(sample, kfm, device, topk=args.topk)
                text = sample["original_instruction"]
            elif mode == "learned_oracle_trend":
                images, topk_local, topk_prob = build_kfm_bundle(sample, kfm, device, topk=args.topk)
                text = sample["instruction"]
            else:
                raise ValueError(f"Unknown policy offline mode: {mode}")

            pred, raw_action = predict_policy_action(
                policy=policy,
                processor=processor,
                text=text,
                images=images,
                device=device,
                dtype=dtype,
            )

            pred_name = ACTION_NAMES.get(pred, str(pred))
            is_correct = int(pred == gt)

            correct += is_correct
            total += 1
            pred_counter[pred] += 1
            gt_counter[gt] += 1

            meta = sample.get("meta", {})

            row = {
                "mode": mode,
                "dataset_index": idx,
                "current_idx": meta.get("current_idx", ""),
                "image_path": meta.get("image_path", ""),
                "parquet_path": meta.get("parquet_path", ""),
                "gt_action": gt,
                "gt_action_name": gt_name,
                "pred_action": pred,
                "pred_action_name": pred_name,
                "correct": is_correct,
                "topk_local": json.dumps(topk_local),
                "topk_prob": json.dumps([float(x) for x in topk_prob]),
                "raw_action": json.dumps(raw_action),
            }

            per_rows.append(row)

            if idx < args.max_visual_cases:
                out_png = mode_vis / f"{mode}_case_{idx:04d}.png"
                save_policy_visual(
                    images=images,
                    title=f"Module 6: OpenFly/LoRA final policy action - {mode}",
                    info_lines=[
                        f"dataset_index={idx}, current_idx={meta.get('current_idx', '')}",
                        f"GT action   = {gt} / {gt_name}",
                        f"Pred action = {pred} / {pred_name}",
                        f"Correct     = {is_correct}",
                        f"topk_local  = {topk_local}",
                        f"topk_prob   = {[round(float(x), 4) for x in topk_prob]}",
                        f"prompt mode = {'with oracle trend text' if mode == 'learned_oracle_trend' else 'no trend text'}",
                    ],
                    out_path=out_png,
                )

        acc = correct / total if total else 0.0
        pred_stop_rate = pred_counter[0] / total if total else 0.0

        summary_by_mode[mode] = {
            "mode": mode,
            "num_samples": total,
            "action_acc": acc,
            "correct": correct,
            "pred_stop_rate": pred_stop_rate,
            "pred_action_distribution": {str(k): int(v) for k, v in sorted(pred_counter.items())},
            "gt_action_distribution": {str(k): int(v) for k, v in sorted(gt_counter.items())},
        }

    per_csv = out / "per_sample.csv"
    if per_rows:
        with open(per_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_rows)

    summary = {
        "module": "openfly_lora_final_policy_offline",
        "note": "This evaluates OpenFly/LoRA predict_action offline on recorded frame bundles. It is not online SR/OSR/SPL.",
        "final_model_dir": args.final_model_dir,
        "model_path": args.model_path,
        "annotation_path": args.annotation_path,
        "parquet_root": args.parquet_root,
        "modes": modes,
        "by_mode": summary_by_mode,
    }

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    index_lines = ["# OpenFly/LoRA final policy offline visualizations", ""]
    for mode in modes:
        index_lines.append(f"## {mode}")
        index_lines.append("")
        for p in sorted((vis_root / mode).glob("*.png")):
            index_lines.append(f"![]({p.relative_to(out).as_posix()})")
            index_lines.append("")
    (out / "policy_visual_index.md").write_text("\n".join(index_lines), encoding="utf-8")

    print("[POLICY OFFLINE] saved:", out / "summary.json")
    print("[POLICY OFFLINE] saved:", per_csv)
    print("[POLICY OFFLINE] visual index:", out / "policy_visual_index.md")


if __name__ == "__main__":
    main()
