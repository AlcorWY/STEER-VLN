#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Make steer_vln imports stable from project root
ROOT = Path(__file__).resolve().parents[1]
STEER_VLN = ROOT / "STEER-VLN"
for p in [str(STEER_VLN), str(ROOT / "code"), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from train_integrated_full import IntegratedOpenFlyDataset
from keyframe.simple_tokenizer import SimpleTextTokenizer
from keyframe.keyframe_scorer import AttentionKeyframeScorer, KeyframeScorerConfig


def load_tokenizer(vocab_path: Path):
    # Compatible with multiple earlier SimpleTextTokenizer variants.
    if hasattr(SimpleTextTokenizer, "load"):
        return SimpleTextTokenizer.load(vocab_path)
    if hasattr(SimpleTextTokenizer, "from_file"):
        return SimpleTextTokenizer.from_file(vocab_path)
    if hasattr(SimpleTextTokenizer, "from_vocab_file"):
        return SimpleTextTokenizer.from_vocab_file(vocab_path)

    data = json.load(open(vocab_path, "r", encoding="utf-8"))
    if isinstance(data, dict) and "token_to_id" in data:
        tok = SimpleTextTokenizer()
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        tok.pad_token_id = tok.token_to_id.get("<pad>", 0)
        return tok

    raise RuntimeError(f"Cannot load tokenizer from {vocab_path}")


def tensor_to_pil_01(x):
    x = x.detach().float().cpu().clamp(0, 1)
    arr = x.permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def tensor_to_pil_norm(x):
    # Dataset history_images/current_image are normalized by the keyframe dataset.
    # Convert roughly back to visible image by min-max per image.
    x = x.detach().float().cpu()
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        y = torch.zeros_like(x)
    else:
        y = (x - mn) / (mx - mn)
    arr = y.permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def masked_softmax(logits, mask):
    logits = logits.masked_fill(~mask.bool(), -1e4)
    return torch.softmax(logits, dim=-1)


def load_kfm(final_model_dir, device):
    ckpt_path = Path(final_model_dir) / "keyframe_scorer_best.pt"
    vocab_path = Path(final_model_dir) / "simple_tokenizer_vocab.json"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("cfg", {})
    tokenizer = load_tokenizer(vocab_path)

    cfg = KeyframeScorerConfig(**cfg_dict)
    model = AttentionKeyframeScorer(cfg, padding_idx=getattr(tokenizer, "pad_token_id", 0))
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, cfg


def make_dataset(args, tokenizer):
    ds_args = SimpleNamespace(
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
        label_temperature=0.7,
        cache_size=32,
    )
    return IntegratedOpenFlyDataset(ds_args, tokenizer)


def draw_label(draw, xy, text, fill=(255, 255, 255), bg=(0, 0, 0)):
    x, y = xy
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=bg)
    draw.text((x, y), text, fill=fill, font=font)


def tile_images(images, labels, border_colors, title, info_lines, out_path):
    cell_w, cell_h = 224, 224
    margin = 12
    label_h = 32
    title_h = 44
    info_h = max(120, 18 * len(info_lines) + 20)

    n = len(images)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))

    W = cols * cell_w + (cols + 1) * margin
    H = title_h + rows * (cell_h + label_h + margin) + info_h

    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    draw_label(draw, (margin, 10), title, fill=(255, 255, 255), bg=(30, 30, 30))

    y0 = title_h
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        x = margin + c * (cell_w + margin)
        y = y0 + r * (cell_h + label_h + margin)

        im = img.resize((cell_w, cell_h))
        canvas.paste(im, (x, y))

        color = border_colors[i]
        for t in range(4):
            draw.rectangle([x - t, y - t, x + cell_w + t, y + cell_h + t], outline=color)

        draw_label(draw, (x, y + cell_h + 4), labels[i], fill=(255, 255, 255), bg=color)

    info_y = title_h + rows * (cell_h + label_h + margin) + 8
    for line in info_lines:
        draw.text((margin, info_y), line, fill=(0, 0, 0))
        info_y += 18

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_model_dir", required=True)
    ap.add_argument("--annotation_path", required=True)
    ap.add_argument("--parquet_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split_name", default="eval_test")
    ap.add_argument("--max_cases", type=int, default=32)
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--max_history", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min_timestep", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--d_scale", type=float, default=12.0)
    ap.add_argument("--z_scale", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfm, tokenizer, cfg = load_kfm(args.final_model_dir, device)

    dataset = make_dataset(args, tokenizer)

    if len(dataset) <= 0:
        report = {
            "split": args.split_name,
            "status": "no_samples",
            "annotation_path": args.annotation_path,
            "parquet_root": args.parquet_root,
        }
        (out_dir / "NO_VISUAL_SAMPLES.md").write_text(
            "# No visual samples\n\n"
            f"- split: {args.split_name}\n"
            f"- annotation_path: {args.annotation_path}\n"
            f"- parquet_root: {args.parquet_root}\n",
            encoding="utf-8",
        )
        (out_dir / "visualization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[VIS][WARN] no visual samples; wrote diagnostic report")
        return

    records = []
    max_cases = min(args.max_cases, len(dataset))

    # Spread cases across dataset instead of only first samples.
    if max_cases <= 1:
        indices = [0]
    else:
        indices = np.linspace(0, len(dataset) - 1, max_cases).round().astype(int).tolist()

    for case_i, idx in enumerate(indices):
        sample = dataset[int(idx)]

        batch = {}
        for k, v in sample.items():
            if torch.is_tensor(v):
                batch[k] = v.unsqueeze(0).to(device)
            else:
                batch[k] = [v] if k in ["instruction", "original_instruction", "trend_text", "meta"] else v

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

        mask = sample["history_mask"].bool()
        valid_idx = torch.nonzero(mask, as_tuple=False).flatten()
        topk = min(3, int(valid_idx.numel()))

        if topk > 0:
            valid_probs = probs[valid_idx]
            top_vals, top_pos = torch.topk(valid_probs, k=topk)
            top_local = valid_idx[top_pos].tolist()
        else:
            top_vals = torch.tensor([])
            top_local = []

        images = []
        labels = []
        colors = []

        # history candidates
        for h in range(sample["history_images"].shape[0]):
            if not bool(sample["history_mask"][h]):
                continue

            img = tensor_to_pil_norm(sample["history_images"][h])
            p = float(probs[h].item())
            is_top = h in top_local
            rank = top_local.index(h) + 1 if is_top else "-"
            color = (220, 40, 40) if is_top else (40, 150, 40)

            images.append(img)
            labels.append(f"H{h} p={p:.3f} rank={rank}")
            colors.append(color)

        # previous/current
        if "previous_raw" in sample:
            images.append(tensor_to_pil_01(sample["previous_raw"]))
            labels.append("previous")
            colors.append((40, 80, 220))

        if "current_raw" in sample:
            images.append(tensor_to_pil_01(sample["current_raw"]))
            labels.append("current")
            colors.append((40, 80, 220))
        else:
            images.append(tensor_to_pil_norm(sample["current_image"]))
            labels.append("current")
            colors.append((40, 80, 220))

        meta = sample.get("meta", {})
        if isinstance(meta, list):
            meta = meta[0]

        info_lines = [
            f"split={args.split_name} sample_index={idx}",
            f"parquet={meta.get('parquet_path', '')}",
            f"current_idx={meta.get('current_idx', '')} candidates={meta.get('candidate_indices', '')}",
            f"topk_local={top_local} topk_prob={[round(float(x), 4) for x in top_vals.tolist()]}",
            f"action_label={int(sample['action_label'].item())} waypoint_raw={[round(float(x), 3) for x in sample['waypoint_raw'].tolist()]}",
            f"stop={float(sample['stop_label'].item()):.1f} prestop={float(sample['prestop_label'].item()):.1f}",
            f"trend={sample.get('trend_text', '')[:180]}",
        ]

        out_png = vis_dir / f"{args.split_name}_case_{case_i:04d}_sample_{int(idx)}.png"
        tile_images(
            images=images,
            labels=labels,
            border_colors=colors,
            title=f"STEER_VLN Offline Module Visualization: {args.split_name}",
            info_lines=info_lines,
            out_path=out_png,
        )

        records.append(
            {
                "case": case_i,
                "dataset_index": int(idx),
                "image": str(out_png),
                "topk_local": top_local,
                "topk_prob": [float(x) for x in top_vals.tolist()],
                "action_label": int(sample["action_label"].item()),
                "stop_label": float(sample["stop_label"].item()),
                "prestop_label": float(sample["prestop_label"].item()),
                "meta": meta,
            }
        )

    report = {
        "split": args.split_name,
        "status": "ok",
        "num_dataset_samples": len(dataset),
        "num_visualized": len(records),
        "records": records,
    }

    (out_dir / "visualization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# STEER_VLN Offline Module Visualization: {args.split_name}", ""]
    for r in records:
        rel = Path(r["image"]).relative_to(out_dir)
        lines.append(f"## Case {r['case']} | sample {r['dataset_index']}")
        lines.append("")
        lines.append(f"![]({rel.as_posix()})")
        lines.append("")
        lines.append(f"- topk_local: {r['topk_local']}")
        lines.append(f"- topk_prob: {[round(float(x), 4) for x in r['topk_prob']]}")
        lines.append("")
    (out_dir / "visualization_index.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[VIS][OK] saved {len(records)} visualizations to {vis_dir}")
    print(f"[VIS][OK] index: {out_dir / 'visualization_index.md'}")


if __name__ == "__main__":
    main()
