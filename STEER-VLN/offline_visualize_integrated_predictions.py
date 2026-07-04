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


def read_csv_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=None):
    try:
        return int(float(x))
    except Exception:
        return default


def get_row(rows, idx):
    if idx < len(rows):
        return rows[idx]
    return {}


def draw_text(draw, xy, text, fill=(0, 0, 0)):
    font = ImageFont.load_default()
    draw.text(xy, text, fill=fill, font=font)


def wrap(text, width=95):
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


def save_current_image_card(current_img, title, sections, out_path):
    W = 1280
    H = 760
    margin = 28
    title_h = 52
    img_size = 360
    line_h = 22

    canvas = Image.new("RGB", (W, H), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.rectangle([0, 0, W, title_h], fill=(30, 30, 30))
    draw.text((margin, 18), title, fill=(255, 255, 255), font=font)

    img = current_img.resize((img_size, img_size))
    x_img = margin
    y_img = title_h + margin
    canvas.paste(img, (x_img, y_img))
    draw.rectangle([x_img, y_img, x_img + img_size, y_img + img_size], outline=(40, 80, 220), width=4)
    draw.text((x_img, y_img + img_size + 8), "Current observation", fill=(0, 0, 0), font=font)

    x = margin + img_size + 42
    y = title_h + margin

    for sec_title, sec_lines in sections:
        draw.rectangle([x - 8, y - 4, W - margin, y + line_h], fill=(225, 225, 225))
        draw.text((x, y), sec_title, fill=(0, 0, 0), font=font)
        y += line_h + 8

        for line in sec_lines:
            for w in wrap(line):
                draw.text((x + 14, y), w, fill=(20, 20, 20), font=font)
                y += line_h
        y += 14

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_model_dir", required=True)
    ap.add_argument("--annotation_path", required=True)
    ap.add_argument("--parquet_root", required=True)
    ap.add_argument("--integrated_per_sample", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split_name", default="eval_test")
    ap.add_argument("--max_cases", type=int, default=32)

    ap.add_argument("--max_history", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min_timestep", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--d_scale", type=float, default=12.0)
    ap.add_argument("--z_scale", type=float, default=5.0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    trend_dir = out / "03_trend_action_predictions"
    wp_dir = out / "04_waypoint_predictions"
    stop_dir = out / "05_stop_prestop_predictions"

    for d in [trend_dir, wp_dir, stop_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(Path(args.final_model_dir) / "simple_tokenizer_vocab.json")
    dataset = make_dataset(args, tokenizer)
    rows = read_csv_rows(args.integrated_per_sample)

    n = min(len(dataset), len(rows))
    if n <= 0:
        raise RuntimeError("No aligned samples found for integrated prediction visualization.")

    max_cases = min(args.max_cases, n)
    indices = np.linspace(0, n - 1, max_cases).round().astype(int).tolist()

    index_lines = [f"# Integrated Prediction Visualizations: {args.split_name}", ""]
    report = {
        "split": args.split_name,
        "num_dataset_samples": len(dataset),
        "num_csv_rows": len(rows),
        "num_visualized": max_cases,
        "cases": [],
    }

    for case_i, idx in enumerate(indices):
        sample = dataset[int(idx)]
        row = get_row(rows, int(idx))

        current_img = tensor_to_pil_01(sample["current_raw"]) if "current_raw" in sample else tensor_to_pil_01(sample["previous_raw"])

        gt_action = int(sample["action_label"].item())
        gt_action_name = ACTION_NAMES.get(gt_action, str(gt_action))

        pred_action = to_int(row.get("trend_aux_action_pred"), None)
        pred_action_name = ACTION_NAMES.get(pred_action, str(pred_action)) if pred_action is not None else "N/A"
        action_correct = row.get("trend_aux_action_correct", "N/A")

        # ---------------- Trend action prediction ----------------
        trend_png = trend_dir / f"{args.split_name}_case_{case_i:04d}_trend_action.png"
        save_current_image_card(
            current_img=current_img,
            title="Module 3: Temporal trend auxiliary action prediction",
            sections=[
                ("Ground truth", [
                    f"action_label = {gt_action}",
                    f"action_name  = {gt_action_name}",
                ]),
                ("Trend head prediction from integrated/eval_test/per_sample.csv", [
                    f"trend_aux_action_pred    = {pred_action}",
                    f"trend_aux_action_name    = {pred_action_name}",
                    f"trend_aux_action_correct = {action_correct}",
                ]),
                ("Trend state text", [
                    str(sample.get("trend_text", "")),
                ]),
                ("Role in final model", [
                    "This head provides auxiliary temporal motion tendency and should not override OpenFly/LoRA final action.",
                ]),
            ],
            out_path=trend_png,
        )

        # ---------------- Waypoint prediction ----------------
        gt_d, gt_yaw, gt_z = [float(x) for x in sample["waypoint_raw"].tolist()]

        pred_d_norm = to_float(row.get("wp_pred_d_norm"), None)
        pred_yaw_norm = to_float(row.get("wp_pred_yaw_norm"), None)
        pred_z_norm = to_float(row.get("wp_pred_z_norm"), None)

        pred_d = pred_d_norm * args.d_scale if pred_d_norm is not None else None
        pred_yaw = pred_yaw_norm * math.pi if pred_yaw_norm is not None else None
        pred_z = pred_z_norm * args.z_scale if pred_z_norm is not None else None

        wp_lines = []
        if pred_d is not None:
            wp_lines.append(f"pred_d   = {pred_d:.3f} m     | error = {abs(pred_d - gt_d):.3f} m")
        else:
            wp_lines.append("pred_d   = N/A")

        if pred_yaw is not None:
            wp_lines.append(
                f"pred_yaw = {pred_yaw:.3f} rad / {pred_yaw * 180.0 / math.pi:.2f} deg"
                f" | error = {abs(pred_yaw - gt_yaw):.3f} rad"
            )
        else:
            wp_lines.append("pred_yaw = N/A")

        if pred_z is not None:
            wp_lines.append(f"pred_z   = {pred_z:.3f} m     | error = {abs(pred_z - gt_z):.3f} m")
        else:
            wp_lines.append("pred_z   = N/A")

        wp_png = wp_dir / f"{args.split_name}_case_{case_i:04d}_waypoint.png"
        save_current_image_card(
            current_img=current_img,
            title="Module 4: Waypoint head prediction",
            sections=[
                ("Ground truth waypoint", [
                    f"gt_d   = {gt_d:.3f} m",
                    f"gt_yaw = {gt_yaw:.3f} rad / {gt_yaw * 180.0 / math.pi:.2f} deg",
                    f"gt_z   = {gt_z:.3f} m",
                ]),
                ("Predicted waypoint", wp_lines),
                ("Raw normalized prediction fields", [
                    f"wp_pred_d_norm   = {row.get('wp_pred_d_norm', 'N/A')}",
                    f"wp_pred_yaw_norm = {row.get('wp_pred_yaw_norm', 'N/A')}",
                    f"wp_pred_z_norm   = {row.get('wp_pred_z_norm', 'N/A')}",
                ]),
            ],
            out_path=wp_png,
        )

        # ---------------- Stop / prestop prediction ----------------
        gt_stop = float(sample["stop_label"].item())
        gt_prestop = float(sample["prestop_label"].item())

        stop_prob = to_float(row.get("stop_prob"), None)
        prestop_prob = to_float(row.get("prestop_prob"), None)

        stop_pred = int(stop_prob >= 0.5) if stop_prob is not None else "N/A"
        prestop_pred = int(prestop_prob >= 0.5) if prestop_prob is not None else "N/A"

        stop_png = stop_dir / f"{args.split_name}_case_{case_i:04d}_stop_prestop.png"
        save_current_image_card(
            current_img=current_img,
            title="Module 5: Stop / prestop temporal terminal prediction",
            sections=[
                ("Ground truth terminal labels", [
                    f"stop_label    = {gt_stop:.1f}",
                    f"prestop_label = {gt_prestop:.1f}",
                ]),
                ("Predicted probabilities", [
                    f"stop_prob    = {stop_prob if stop_prob is not None else 'N/A'}",
                    f"stop_pred    = {stop_pred}",
                    f"prestop_prob = {prestop_prob if prestop_prob is not None else 'N/A'}",
                    f"prestop_pred = {prestop_pred}",
                ]),
                ("Role in final model", [
                    "Stop/prestop is an internal temporal terminal trend state, not an external evaluation-time stop override.",
                ]),
            ],
            out_path=stop_png,
        )

        report["cases"].append({
            "case": case_i,
            "dataset_index": int(idx),
            "trend_png": str(trend_png),
            "waypoint_png": str(wp_png),
            "stop_png": str(stop_png),
            "gt_action": gt_action,
            "pred_action": pred_action,
            "action_correct": action_correct,
            "gt_waypoint": [gt_d, gt_yaw, gt_z],
            "pred_waypoint": [pred_d, pred_yaw, pred_z],
            "stop_prob": stop_prob,
            "prestop_prob": prestop_prob,
        })

    index_lines.append("## Module 3: Temporal trend auxiliary action prediction")
    index_lines.append("")
    for p in sorted(trend_dir.glob("*.png")):
        index_lines.append(f"![]({p.relative_to(out).as_posix()})")
        index_lines.append("")

    index_lines.append("## Module 4: Waypoint head prediction")
    index_lines.append("")
    for p in sorted(wp_dir.glob("*.png")):
        index_lines.append(f"![]({p.relative_to(out).as_posix()})")
        index_lines.append("")

    index_lines.append("## Module 5: Stop / Prestop prediction")
    index_lines.append("")
    for p in sorted(stop_dir.glob("*.png")):
        index_lines.append(f"![]({p.relative_to(out).as_posix()})")
        index_lines.append("")

    (out / "integrated_prediction_visual_index.md").write_text("\n".join(index_lines), encoding="utf-8")
    (out / "integrated_prediction_visual_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] integrated prediction visualizations saved:", out)
    print("[OK] index:", out / "integrated_prediction_visual_index.md")


if __name__ == "__main__":
    main()
