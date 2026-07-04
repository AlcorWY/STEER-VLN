#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
STEER_VLN = ROOT / "STEER-VLN"

for p in [str(STEER_VLN), str(ROOT / "code"), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from train_integrated_full import IntegratedOpenFlyDataset
from keyframe.simple_tokenizer import SimpleTextTokenizer
from keyframe.keyframe_scorer import AttentionKeyframeScorer, KeyframeScorerConfig


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


def load_tokenizer(vocab_path: Path):
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
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        tok.pad_token_id = tok.token_to_id.get("<pad>", 0)
        return tok

    if isinstance(data, dict):
        tok.token_to_id = data
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
    final_model_dir = Path(final_model_dir)
    ckpt_path = final_model_dir / "keyframe_scorer_best.pt"
    vocab_path = final_model_dir / "simple_tokenizer_vocab.json"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("cfg", {})
    tokenizer = load_tokenizer(vocab_path)

    cfg = KeyframeScorerConfig(**cfg_dict)
    model = AttentionKeyframeScorer(cfg, padding_idx=getattr(tokenizer, "pad_token_id", 0))

    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer


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


def load_integrated_csv(path):
    path = Path(path)
    if not path.exists():
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def get_metric_row(rows, dataset_index, case_i):
    if not rows:
        return {}

    # Prefer explicit index fields if available.
    for key in ["dataset_index", "sample_index", "idx", "index"]:
        if key in rows[0]:
            for r in rows:
                try:
                    if int(float(r[key])) == int(dataset_index):
                        return r
                except Exception:
                    pass

    if int(dataset_index) < len(rows):
        return rows[int(dataset_index)]

    if case_i < len(rows):
        return rows[case_i]

    return {}


def draw_label(draw, xy, text, fill=(255, 255, 255), bg=(20, 20, 20)):
    x, y = xy
    font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3], fill=bg)
    draw.text((x, y), text, fill=fill, font=font)


def wrap_text(text, width=92):
    text = str(text)
    out = []
    while len(text) > width:
        cut = text.rfind(" ", 0, width)
        if cut <= 0:
            cut = width
        out.append(text[:cut])
        text = text[cut:].strip()
    if text:
        out.append(text)
    return out


def save_grid_image(images, labels, colors, title, info_lines, out_path, cols=4):
    cell = 224
    margin = 14
    label_h = 30
    title_h = 46

    n = len(images)
    cols = min(cols, max(1, n))
    rows = int(math.ceil(n / cols))
    info_lines = [str(x) for x in info_lines]
    info_h = max(90, 18 * len(info_lines) + 20)

    W = cols * cell + (cols + 1) * margin
    H = title_h + rows * (cell + label_h + margin) + info_h

    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    draw_label(draw, (margin, 12), title, bg=(30, 30, 30))

    y0 = title_h
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        x = margin + c * (cell + margin)
        y = y0 + r * (cell + label_h + margin)

        im = img.resize((cell, cell))
        canvas.paste(im, (x, y))

        color = colors[i]
        for t in range(4):
            draw.rectangle([x - t, y - t, x + cell + t, y + cell + t], outline=color)

        draw_label(draw, (x, y + cell + 5), labels[i], bg=color)

    info_y = title_h + rows * (cell + label_h + margin) + 8
    for line in info_lines:
        draw.text((margin, info_y), line, fill=(0, 0, 0))
        info_y += 18

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_text_card(title, sections, out_path, width=1120):
    margin = 28
    line_h = 22
    title_h = 48

    lines = []
    for sec_title, sec_lines in sections:
        lines.append(("section", sec_title))
        for line in sec_lines:
            for w in wrap_text(line, width=115):
                lines.append(("text", w))
        lines.append(("space", ""))

    H = title_h + margin * 2 + line_h * len(lines)
    canvas = Image.new("RGB", (width, H), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.rectangle([0, 0, width, title_h], fill=(30, 30, 30))
    draw.text((margin, 16), title, fill=(255, 255, 255), font=font)

    y = title_h + margin
    for kind, text in lines:
        if kind == "section":
            draw.rectangle([margin - 6, y - 3, width - margin, y + line_h - 3], fill=(225, 225, 225))
            draw.text((margin, y), text, fill=(0, 0, 0), font=font)
        elif kind == "text":
            draw.text((margin + 14, y), text, fill=(20, 20, 20), font=font)
        y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def metric_get(row, keys, default="N/A"):
    for k in keys:
        if k in row and row[k] not in ["", None]:
            return row[k]
    return default


def compute_kfm_outputs(kfm, sample, device, topk=3):
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

    mask = sample["history_mask"].bool()
    valid_idx = torch.nonzero(mask, as_tuple=False).flatten()

    topk = min(topk, int(valid_idx.numel()))
    if topk > 0:
        valid_probs = probs[valid_idx]
        top_vals, top_pos = torch.topk(valid_probs, k=topk)
        top_local = valid_idx[top_pos].tolist()
        weights = top_vals / top_vals.sum().clamp_min(1e-8)
    else:
        top_vals = torch.tensor([])
        top_local = []
        weights = torch.tensor([])

    return probs, top_local, top_vals, weights


def build_fused_keyframe(sample, top_local, weights):
    if "history_raw" in sample and len(top_local) > 0:
        fused = torch.zeros_like(sample["history_raw"][top_local[0]])
        for w, idx in zip(weights, top_local):
            fused = fused + float(w) * sample["history_raw"][idx]
        return fused.clamp(0, 1)

    if "previous_raw" in sample:
        return sample["previous_raw"]

    return sample["current_image"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_model_dir", required=True)
    ap.add_argument("--annotation_path", required=True)
    ap.add_argument("--parquet_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split_name", default="eval_test")
    ap.add_argument("--integrated_per_sample", default="")
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

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    module_dirs = {
        "keyframe": out / "01_keyframe_selection",
        "bundle": out / "02_frame_bundle",
        "trend": out / "03_trend_action",
        "waypoint": out / "04_waypoint_head",
        "stop": out / "05_stop_prestop",
    }

    for d in module_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfm, tokenizer = load_kfm(args.final_model_dir, device)
    dataset = make_dataset(args, tokenizer)

    if len(dataset) <= 0:
        raise RuntimeError("No samples found for module visualization.")

    integrated_rows = load_integrated_csv(args.integrated_per_sample) if args.integrated_per_sample else []

    max_cases = min(args.max_cases, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, max_cases).round().astype(int).tolist()

    index_md = {
        "keyframe": [],
        "bundle": [],
        "trend": [],
        "waypoint": [],
        "stop": [],
    }

    report = {
        "split": args.split_name,
        "num_dataset_samples": len(dataset),
        "num_visualized_cases": max_cases,
        "modules": {k: str(v) for k, v in module_dirs.items()},
        "cases": [],
    }

    for case_i, ds_idx in enumerate(indices):
        sample = dataset[int(ds_idx)]
        row = get_metric_row(integrated_rows, ds_idx, case_i)

        probs, top_local, top_vals, weights = compute_kfm_outputs(kfm, sample, device, topk=3)
        fused = build_fused_keyframe(sample, top_local, weights)

        meta = sample.get("meta", {})
        action_label = int(sample["action_label"].item())
        action_name = ACTION_NAMES.get(action_label, str(action_label))
        waypoint_raw = [float(x) for x in sample["waypoint_raw"].tolist()]
        stop_label = float(sample["stop_label"].item())
        prestop_label = float(sample["prestop_label"].item())

        # ----------------------------------------------------
        # 01 Keyframe scorer visualization
        # ----------------------------------------------------
        k_imgs, k_labels, k_colors = [], [], []

        for h in range(sample["history_images"].shape[0]):
            if not bool(sample["history_mask"][h]):
                continue

            if "history_raw" in sample:
                img = tensor_to_pil_01(sample["history_raw"][h])
            else:
                img = tensor_to_pil_norm(sample["history_images"][h])

            p = float(probs[h])
            is_top = h in top_local
            rank = top_local.index(h) + 1 if is_top else "-"
            color = (210, 40, 40) if is_top else (40, 150, 40)

            k_imgs.append(img)
            k_labels.append(f"H{h} p={p:.3f} rank={rank}")
            k_colors.append(color)

        k_out = module_dirs["keyframe"] / f"{args.split_name}_case_{case_i:04d}_keyframe.png"
        save_grid_image(
            k_imgs,
            k_labels,
            k_colors,
            title="Module 1: Keyframe Scorer - historical frame scoring",
            info_lines=[
                f"sample={ds_idx}, current_idx={meta.get('current_idx', '')}",
                f"topk_local={top_local}, topk_prob={[round(float(x), 4) for x in top_vals.tolist()]}",
                "red = selected top-k keyframe candidates; green = valid history candidates",
            ],
            out_path=k_out,
            cols=4,
        )
        index_md["keyframe"].append(k_out)

        # ----------------------------------------------------
        # 02 Frame bundle visualization
        # ----------------------------------------------------
        b_imgs = [
            tensor_to_pil_01(fused) if fused.ndim == 3 else tensor_to_pil_norm(fused),
            tensor_to_pil_01(sample["previous_raw"]) if "previous_raw" in sample else tensor_to_pil_norm(sample["current_image"]),
            tensor_to_pil_01(sample["current_raw"]) if "current_raw" in sample else tensor_to_pil_norm(sample["current_image"]),
        ]
        b_labels = [
            "weighted keyframe",
            "previous frame",
            "current frame",
        ]
        b_colors = [
            (210, 40, 40),
            (40, 80, 220),
            (40, 80, 220),
        ]

        b_out = module_dirs["bundle"] / f"{args.split_name}_case_{case_i:04d}_bundle.png"
        save_grid_image(
            b_imgs,
            b_labels,
            b_colors,
            title="Module 2: Temporal frame bundle construction",
            info_lines=[
                "Input to OpenFly/LoRA is organized as [weighted keyframe, previous, current].",
                f"topk_local={top_local}, weights={[round(float(x), 4) for x in weights.tolist()]}",
            ],
            out_path=b_out,
            cols=3,
        )
        index_md["bundle"].append(b_out)

        # ----------------------------------------------------
        # 03 Trend action head visualization
        # ----------------------------------------------------
        trend_pred = metric_get(row, ["action_pred", "pred_action", "trend_action_pred", "aux_action_pred"])
        trend_acc = metric_get(row, ["action_correct", "correct", "action_acc"])

        t_out = module_dirs["trend"] / f"{args.split_name}_case_{case_i:04d}_trend_action.png"
        save_text_card(
            title="Module 3: Temporal trend / auxiliary action head",
            sections=[
                ("Ground truth action", [
                    f"action_label = {action_label}",
                    f"action_name  = {action_name}",
                ]),
                ("Auxiliary trend prediction", [
                    f"pred_action = {trend_pred}",
                    f"correct     = {trend_acc}",
                ]),
                ("Trend state text", [
                    str(sample.get("trend_text", "")),
                ]),
                ("Interpretation", [
                    "This module provides auxiliary temporal state; it should not override OpenFly/LoRA final action.",
                ]),
            ],
            out_path=t_out,
        )
        index_md["trend"].append(t_out)

        # ----------------------------------------------------
        # 04 Waypoint head visualization
        # ----------------------------------------------------
        wp_keys = [
            "mae_d", "mae_yaw", "mae_z", "norm_l1", "yaw_sign_acc",
            "waypoint_pred", "wp_pred", "pred_d", "pred_yaw", "pred_z",
            "target_d", "target_yaw", "target_z",
        ]
        wp_lines = []
        for k in wp_keys:
            if k in row:
                wp_lines.append(f"{k} = {row[k]}")

        if not wp_lines:
            wp_lines.append("No per-sample waypoint prediction columns found in integrated per_sample.csv.")

        w_out = module_dirs["waypoint"] / f"{args.split_name}_case_{case_i:04d}_waypoint.png"
        save_text_card(
            title="Module 4: Waypoint head - distance / yaw / vertical prediction",
            sections=[
                ("Ground truth waypoint", [
                    f"d   = {waypoint_raw[0]:.3f} m",
                    f"yaw = {waypoint_raw[1]:.3f} rad / {waypoint_raw[1] * 180.0 / math.pi:.2f} deg",
                    f"z   = {waypoint_raw[2]:.3f} m",
                ]),
                ("Prediction and error fields from offline eval", wp_lines),
                ("Interpretation", [
                    "This module estimates local motion trend. It is an auxiliary prediction head for module analysis.",
                ]),
            ],
            out_path=w_out,
        )
        index_md["waypoint"].append(w_out)

        # ----------------------------------------------------
        # 05 Stop / prestop visualization
        # ----------------------------------------------------
        stop_keys = [
            "stop_prob", "stop_pred", "stop_label", "stop_correct",
            "prestop_prob", "prestop_pred", "prestop_label", "prestop_correct",
            "stop_logit", "prestop_logit",
        ]
        stop_lines = []
        for k in stop_keys:
            if k in row:
                stop_lines.append(f"{k} = {row[k]}")

        if not stop_lines:
            stop_lines.append("No per-sample stop/prestop probability columns found in integrated per_sample.csv.")

        s_out = module_dirs["stop"] / f"{args.split_name}_case_{case_i:04d}_stop_prestop.png"
        save_text_card(
            title="Module 5: Stop / prestop temporal terminal state",
            sections=[
                ("Ground truth terminal state", [
                    f"stop_label    = {stop_label:.1f}",
                    f"prestop_label = {prestop_label:.1f}",
                ]),
                ("Prediction fields from offline eval", stop_lines),
                ("Interpretation", [
                    "Stop/prestop is treated as temporal terminal trend state, not as an external eval override rule.",
                ]),
            ],
            out_path=s_out,
        )
        index_md["stop"].append(s_out)

        report["cases"].append(
            {
                "case": case_i,
                "dataset_index": int(ds_idx),
                "action_label": action_label,
                "action_name": action_name,
                "topk_local": top_local,
                "topk_prob": [float(x) for x in top_vals.tolist()],
                "keyframe_fig": str(k_out),
                "bundle_fig": str(b_out),
                "trend_fig": str(t_out),
                "waypoint_fig": str(w_out),
                "stop_fig": str(s_out),
            }
        )

    # Write module index markdown
    lines = [f"# STEER_VLN Offline Module Visualization by Category: {args.split_name}", ""]
    module_titles = [
        ("keyframe", "Module 1: Keyframe Scorer"),
        ("bundle", "Module 2: Temporal Frame Bundle"),
        ("trend", "Module 3: Temporal Trend / Action Head"),
        ("waypoint", "Module 4: Waypoint Head"),
        ("stop", "Module 5: Stop / Prestop Head"),
    ]

    for key, title in module_titles:
        lines.append(f"## {title}")
        lines.append("")
        for p in index_md[key]:
            rel = p.relative_to(out)
            lines.append(f"![]({rel.as_posix()})")
            lines.append("")
        lines.append("")

    (out / "module_visual_index.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "module_visual_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] categorized module visualizations generated")
    print("[OUT]", out)
    print("[INDEX]", out / "module_visual_index.md")


if __name__ == "__main__":
    main()
