# ===== STEER_VLN isolated import path =====
import sys as _steer_vln_sys
from pathlib import Path as _STEER_VLNPath

_STEER_VLN_FILE = _STEER_VLNPath(__file__).resolve()
if _STEER_VLN_FILE.parent.name == "keyframe":
    _STEER_VLN_DIR = _STEER_VLN_FILE.parents[1]
else:
    _STEER_VLN_DIR = _STEER_VLN_FILE.parent

_STEER_VLN_ROOT = _STEER_VLN_DIR.parents[0]

for _p in [str(_STEER_VLN_DIR), str(_STEER_VLN_ROOT / "code"), str(_STEER_VLN_ROOT)]:
    if _p not in _steer_vln_sys.path:
        _steer_vln_sys.path.insert(0, _p)
# ===== end STEER_VLN isolated import path =====

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from keyframe.keyframe_scorer import AttentionKeyframeScorer, KeyframeScorerConfig
from keyframe.simple_tokenizer import SimpleTextTokenizer
from keyframe.parquet_keyframe_dataset import (
    action_type_value_to_id,
    pil_to_tensor,
    normalize_image_obj,
)


TURN_ACTION_IDS = {2, 3, 4, 5, 6, 7}


def wrap_angle(x: float) -> float:
    while x > math.pi:
        x -= 2 * math.pi
    while x < -math.pi:
        x += 2 * math.pi
    return x


def motion_feat(rows: List[Dict[str, Any]], cand_idx: int, cur_idx: int) -> List[float]:
    pc = np.asarray(rows[cur_idx]["pos"], dtype=np.float32)
    ph = np.asarray(rows[cand_idx]["pos"], dtype=np.float32)

    dx, dy, dz = (pc - ph).tolist()
    dyaw = wrap_angle(float(rows[cur_idx]["yaw"]) - float(rows[cand_idx]["yaw"]))

    return [
        float(dx / 50.0),
        float(dy / 50.0),
        float(dz / 50.0),
        float(dyaw / math.pi),
    ]


def load_rows(parquet_path: Path) -> List[Dict[str, Any]]:
    ds = load_dataset("parquet", data_files=str(parquet_path), split="train")
    rows = list(ds)
    rows = sorted(rows, key=lambda r: int(r["frame_index"]))
    return rows


def safe_action_ids(rows: List[Dict[str, Any]]) -> List[int]:
    ids = []
    for r in rows:
        try:
            ids.append(action_type_value_to_id(r["action_type"], r["action_value"]))
        except Exception:
            # 保守兜底：未知动作按 forward 处理，避免 eval 中断。
            ids.append(1)
    return ids


def action_name(action_id: int) -> str:
    names = {
        0: "STOP",
        1: "FWD3",
        2: "LEFT",
        3: "RIGHT",
        4: "UP",
        5: "DOWN",
        6: "MOVE_L",
        7: "MOVE_R",
        8: "FWD6",
        9: "FWD9",
    }
    return names.get(int(action_id), f"UNK{action_id}")


def get_reference_points(action_ids: List[int]) -> Dict[str, List[int]]:
    change_points = []
    turn_points = []
    prestop_points = []

    for i in range(1, len(action_ids)):
        if action_ids[i] != action_ids[i - 1]:
            change_points.append(i)

    for i, a in enumerate(action_ids):
        if a in TURN_ACTION_IDS:
            turn_points.append(i)

        if a == 0 and i > 0:
            prestop_points.append(i - 1)

    return {
        "change": sorted(set(change_points)),
        "turn": sorted(set(turn_points)),
        "prestop": sorted(set(prestop_points)),
    }


def adjust_reference_points_for_current(
    ref: Dict[str, List[int]],
    current_idx: int,
) -> Dict[str, List[int]]:
    """
    Eval-time adjustment for PRESTOP.

    In trajectory-level reference construction, PRESTOP can equal current_idx.
    But keyframe scorer only selects from historical candidates: [0, current_idx).

    If PRESTOP contains current_idx, also count current_idx - 1 as a valid
    PRESTOP reference point. This avoids underestimating prestop hits.
    """
    adjusted = {
        "change": list(ref.get("change", [])),
        "turn": list(ref.get("turn", [])),
        "prestop": list(ref.get("prestop", [])),
    }

    if current_idx in adjusted["prestop"] and current_idx > 0:
        adjusted["prestop"].append(current_idx - 1)

    adjusted["change"] = sorted(set(adjusted["change"]))
    adjusted["turn"] = sorted(set(adjusted["turn"]))
    adjusted["prestop"] = sorted(set(adjusted["prestop"]))

    return adjusted


def nearest_distance(x: Optional[int], points: List[int]) -> Optional[int]:
    if x is None or not points:
        return None
    return int(min(abs(x - p) for p in points))


def in_window(x: Optional[int], points: List[int], window: int) -> bool:
    if x is None:
        return False
    return any(abs(x - p) <= window for p in points)


def build_eval_sample(
    rows: List[Dict[str, Any]],
    cur_idx: int,
    max_history: int,
    image_size: int,
    tokenizer: SimpleTextTokenizer,
    instruction: str,
    exclude_previous: bool = True,
) -> Tuple[Dict[str, torch.Tensor], List[Optional[int]]]:
    # Default candidate range excludes previous frame cur_idx - 1.
    # This matches the runtime bundle [keyframe, previous, current].
    end_exclusive = cur_idx - 1 if exclude_previous else cur_idx
    start = max(0, end_exclusive - max_history)
    candidate_indices = list(range(start, end_exclusive))

    if len(candidate_indices) == 0:
        # Early-step fallback: keep one candidate to avoid an invalid all-pad sample.
        # Runtime closed-loop also falls back to residual input when history is too short.
        candidate_indices = [max(0, cur_idx - 1)]

    valid_candidate_indices = list(candidate_indices)

    history_tensors = []
    motion_feats = []
    history_mask = []

    for j in candidate_indices:
        history_tensors.append(pil_to_tensor(rows[j]["image"], image_size))
        motion_feats.append(motion_feat(rows, j, cur_idx))
        history_mask.append(True)

    candidate_with_pad: List[Optional[int]] = list(candidate_indices)

    while len(history_tensors) < max_history:
        history_tensors.insert(0, torch.zeros(3, image_size, image_size))
        motion_feats.insert(0, [0.0, 0.0, 0.0, 0.0])
        history_mask.insert(0, False)
        candidate_with_pad.insert(0, None)

    if len(history_tensors) > max_history:
        history_tensors = history_tensors[-max_history:]
        motion_feats = motion_feats[-max_history:]
        history_mask = history_mask[-max_history:]
        candidate_with_pad = candidate_with_pad[-max_history:]

    encoded = tokenizer(
        instruction,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        padding=False,
        return_attention_mask=True,
    )

    batch = {
        "history_images": torch.stack(history_tensors, dim=0).unsqueeze(0),
        "current_image": pil_to_tensor(rows[cur_idx]["image"], image_size).unsqueeze(0),
        "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long).unsqueeze(0),
        "motion_feats": torch.tensor(motion_feats, dtype=torch.float32).unsqueeze(0),
        "history_mask": torch.tensor(history_mask, dtype=torch.bool).unsqueeze(0),
    }

    return batch, candidate_with_pad


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def predict_scores(
    model: AttentionKeyframeScorer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        logits = model(
            history_images=batch["history_images"],
            current_image=batch["current_image"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            motion_feats=batch["motion_feats"],
            history_mask=batch["history_mask"],
        )

        logits = logits[0]
        mask = batch["history_mask"][0]

        logits = logits.masked_fill(~mask.bool(), -1e4)
        probs = torch.softmax(logits, dim=-1)

    return logits.detach().cpu().numpy(), probs.detach().cpu().numpy()


def get_topk_candidates(
    probs: np.ndarray,
    candidate_with_pad: List[Optional[int]],
    k: int = 2,
) -> List[Tuple[Optional[int], float, int]]:
    pairs = []
    for local_i, cand_idx in enumerate(candidate_with_pad):
        if cand_idx is None:
            continue
        pairs.append((cand_idx, float(probs[local_i]), local_i))

    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    return pairs[:k]


def pil_to_vis_array(img_obj) -> np.ndarray:
    img = normalize_image_obj(img_obj)
    return np.asarray(img)


def flags_for_idx(idx: Optional[int], ref: Dict[str, List[int]], top_rank: Optional[int] = None) -> str:
    if idx is None:
        return "PAD"

    flags = []
    if top_rank == 1:
        flags.append("TOP1")
    elif top_rank == 2:
        flags.append("TOP2")

    if idx in ref["change"]:
        flags.append("CHG")
    if idx in ref["turn"]:
        flags.append("TURN")
    if idx in ref["prestop"]:
        flags.append("PRESTOP")

    return "|".join(flags) if flags else "-"


def visualize_step(
    rows: List[Dict[str, Any]],
    action_ids: List[int],
    candidate_with_pad: List[Optional[int]],
    probs: np.ndarray,
    cur_idx: int,
    top_items: List[Tuple[Optional[int], float, int]],
    ref: Dict[str, List[int]],
    out_file: Path,
    title_prefix: str,
):
    max_history = len(candidate_with_pad)
    ncols = max_history + 1

    fig, axes = plt.subplots(1, ncols, figsize=(2.7 * ncols, 3.6))

    if ncols == 1:
        axes = [axes]

    top_rank_by_idx = {}
    for rank, (cand_idx, _, _) in enumerate(top_items, start=1):
        top_rank_by_idx[cand_idx] = rank

    for local_i, cand_idx in enumerate(candidate_with_pad):
        ax = axes[local_i]
        ax.axis("off")

        if cand_idx is None:
            ax.set_title("PAD", fontsize=9)
            continue

        img = pil_to_vis_array(rows[cand_idx]["image"])
        ax.imshow(img)

        rank = top_rank_by_idx.get(cand_idx, None)
        flags = flags_for_idx(cand_idx, ref, rank)
        prob = float(probs[local_i])

        act = action_name(action_ids[cand_idx])
        ax.set_title(
            f"cand {cand_idx}\n{act} p={prob:.3f}\n{flags}",
            fontsize=8,
        )

    ax = axes[-1]
    ax.axis("off")
    ax.imshow(pil_to_vis_array(rows[cur_idx]["image"]))
    ax.set_title(
        f"CURRENT {cur_idx}\n{action_name(action_ids[cur_idx])}",
        fontsize=9,
    )

    top1 = top_items[0][0] if len(top_items) > 0 else None
    top2 = top_items[1][0] if len(top_items) > 1 else None

    subtitle = (
        f"{title_prefix} | cur={cur_idx} act={action_name(action_ids[cur_idx])} | "
        f"top1={top1}, top2={top2} | "
        f"CHG={ref['change'][:8]} TURN={ref['turn'][:8]} PRESTOP={ref['prestop'][:8]}"
    )
    fig.suptitle(subtitle, fontsize=10)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out_file, dpi=140)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_path", type=str, default="dataset/Annotation/train_airsim16_high.json")
    parser.add_argument("--parquet_root", type=str, default="dataset/hf_openfly_airsim16/traj")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="runs/eval_keyframe_scorer")

    parser.add_argument("--max_episodes", type=int, default=20)
    parser.add_argument("--max_steps_per_episode", type=int, default=12)
    parser.add_argument("--step_stride", type=int, default=4)
    parser.add_argument("--start_idx", type=int, default=1)

    parser.add_argument("--max_history", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--categories", nargs="*", default=None)

    parser.add_argument("--change_window", type=int, default=2)
    parser.add_argument("--turn_window", type=int, default=3)
    parser.add_argument("--prestop_window", type=int, default=3)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_visualize", action="store_true")

    # Default: exclude previous frame from keyframe candidates.
    parser.add_argument("--include_previous_in_candidates", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if args.vocab_path is None:
        vocab_path = checkpoint_path.parent / "simple_tokenizer_vocab.json"
    else:
        vocab_path = Path(args.vocab_path)

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Cannot find vocab file: {vocab_path}. "
            f"Pass --vocab_path or use the checkpoint directory that contains simple_tokenizer_vocab.json."
        )

    tokenizer = SimpleTextTokenizer.load(vocab_path)

    cfg_dict = dict(ckpt["cfg"])
    if args.max_history is not None:
        cfg_dict["max_history"] = args.max_history
    if args.image_size is not None:
        cfg_dict["image_size"] = args.image_size

    # 保持和训练 tokenizer 对齐。
    if cfg_dict.get("vocab_size", len(tokenizer)) != len(tokenizer):
        print(
            f"[WARN] checkpoint cfg vocab_size={cfg_dict.get('vocab_size')} "
            f"but loaded vocab size={len(tokenizer)}. Use loaded vocab size."
        )
        cfg_dict["vocab_size"] = len(tokenizer)

    cfg = KeyframeScorerConfig(**cfg_dict)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    model = AttentionKeyframeScorer(cfg, padding_idx=tokenizer.pad_token_id)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    annotation = json.load(open(args.annotation_path, "r", encoding="utf-8"))
    parquet_root = Path(args.parquet_root)

    if args.categories:
        keep = set(args.categories)
        annotation = [
            x for x in annotation
            if len(x.get("image_path", "").split("/")) > 2
            and x["image_path"].split("/")[2] in keep
        ]

    usable = []
    for item in annotation:
        image_path = item["image_path"]
        parquet_path = parquet_root / Path(image_path).with_suffix(".parquet")
        if parquet_path.exists():
            usable.append((item, parquet_path))

    if args.max_episodes and args.max_episodes > 0:
        usable = usable[:args.max_episodes]

    print(f"[Eval] checkpoint={checkpoint_path}")
    print(f"[Eval] vocab={vocab_path}")
    print(f"[Eval] usable episodes={len(usable)}")
    exclude_previous = not args.include_previous_in_candidates
    print(f"[Eval] output_dir={output_dir}")
    print(f"[Eval] exclude_previous={exclude_previous}")

    csv_path = output_dir / "summary.csv"
    jsonl_path = output_dir / "summary.jsonl"

    csv_fields = [
        "episode_i",
        "image_path",
        "parquet_path",
        "current_idx",
        "current_action",
        "top1_idx",
        "top1_prob",
        "top1_flags",
        "top2_idx",
        "top2_prob",
        "top2_flags",
        "nearest_change_dist",
        "nearest_turn_dist",
        "nearest_prestop_dist",
        "hit_change_window",
        "hit_turn_window",
        "hit_prestop_window",
        "figure",
    ]

    rows_for_csv = []

    total_steps = 0
    hit_change = 0
    hit_turn = 0
    hit_prestop = 0

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for episode_i, (item, parquet_path) in enumerate(tqdm(usable, desc="eval episodes")):
            try:
                rows = load_rows(parquet_path)
            except Exception as e:
                print(f"[WARN] failed to load {parquet_path}: {e}")
                continue

            if len(rows) < 2:
                continue

            action_ids = safe_action_ids(rows)
            ref = get_reference_points(action_ids)

            instruction = str(item.get("gpt_instruction", "")).lower()
            image_path = item["image_path"]

            effective_start_idx = max(args.start_idx, 2) if exclude_previous else args.start_idx
            step_indices = list(range(effective_start_idx, len(rows), args.step_stride))
            if args.max_steps_per_episode and args.max_steps_per_episode > 0:
                step_indices = step_indices[:args.max_steps_per_episode]

            for cur_idx in step_indices:
                batch, candidate_with_pad = build_eval_sample(
                    rows=rows,
                    cur_idx=cur_idx,
                    max_history=cfg.max_history,
                    image_size=cfg.image_size,
                    tokenizer=tokenizer,
                    instruction=instruction,
                    exclude_previous=exclude_previous,
                )

                logits, probs = predict_scores(model, batch, device)
                top_items = get_topk_candidates(probs, candidate_with_pad, k=2)

                # Adjust PRESTOP for current_idx:
                # if raw PRESTOP contains current_idx, current_idx - 1 is also valid.
                eval_ref = adjust_reference_points_for_current(ref, cur_idx)

                top1_idx = top_items[0][0] if len(top_items) > 0 else None
                top1_prob = top_items[0][1] if len(top_items) > 0 else None
                top2_idx = top_items[1][0] if len(top_items) > 1 else None
                top2_prob = top_items[1][1] if len(top_items) > 1 else None

                top1_flags = flags_for_idx(top1_idx, eval_ref, top_rank=1)
                top2_flags = flags_for_idx(top2_idx, eval_ref, top_rank=2)

                nearest_change = nearest_distance(top1_idx, eval_ref["change"])
                nearest_turn = nearest_distance(top1_idx, eval_ref["turn"])
                nearest_prestop = nearest_distance(top1_idx, eval_ref["prestop"])

                h_change = in_window(top1_idx, eval_ref["change"], args.change_window)
                h_turn = in_window(top1_idx, eval_ref["turn"], args.turn_window)
                h_prestop = in_window(top1_idx, eval_ref["prestop"], args.prestop_window)

                total_steps += 1
                hit_change += int(h_change)
                hit_turn += int(h_turn)
                hit_prestop += int(h_prestop)

                fig_rel = ""
                if not args.no_visualize:
                    safe_name = image_path.replace("/", "__")
                    fig_file = vis_dir / f"ep{episode_i:04d}__t{cur_idx:04d}__{safe_name}.png"
                    visualize_step(
                        rows=rows,
                        action_ids=action_ids,
                        candidate_with_pad=candidate_with_pad,
                        probs=probs,
                        cur_idx=cur_idx,
                        top_items=top_items,
                        ref=eval_ref,
                        out_file=fig_file,
                        title_prefix=f"ep={episode_i} {image_path}",
                    )
                    fig_rel = str(fig_file.relative_to(output_dir))

                record = {
                    "episode_i": episode_i,
                    "image_path": image_path,
                    "parquet_path": str(parquet_path),
                    "current_idx": cur_idx,
                    "current_action": action_name(action_ids[cur_idx]),
                    "candidate_indices": candidate_with_pad,
                    "candidate_probs": [float(x) for x in probs.tolist()],
                    "top1_idx": top1_idx,
                    "top1_prob": top1_prob,
                    "top1_flags": top1_flags,
                    "top2_idx": top2_idx,
                    "top2_prob": top2_prob,
                    "top2_flags": top2_flags,
                    "change_points": ref["change"],
                    "turn_points": ref["turn"],
                    "prestop_points_raw": ref["prestop"],
                    "prestop_points_eval": eval_ref["prestop"],
                    "nearest_change_dist": nearest_change,
                    "nearest_turn_dist": nearest_turn,
                    "nearest_prestop_dist": nearest_prestop,
                    "hit_change_window": h_change,
                    "hit_turn_window": h_turn,
                    "hit_prestop_window": h_prestop,
                    "figure": fig_rel,
                }

                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

                rows_for_csv.append({
                    "episode_i": episode_i,
                    "image_path": image_path,
                    "parquet_path": str(parquet_path),
                    "current_idx": cur_idx,
                    "current_action": action_name(action_ids[cur_idx]),
                    "top1_idx": top1_idx,
                    "top1_prob": "" if top1_prob is None else f"{top1_prob:.6f}",
                    "top1_flags": top1_flags,
                    "top2_idx": top2_idx,
                    "top2_prob": "" if top2_prob is None else f"{top2_prob:.6f}",
                    "top2_flags": top2_flags,
                    "nearest_change_dist": nearest_change,
                    "nearest_turn_dist": nearest_turn,
                    "nearest_prestop_dist": nearest_prestop,
                    "hit_change_window": int(h_change),
                    "hit_turn_window": int(h_turn),
                    "hit_prestop_window": int(h_prestop),
                    "figure": fig_rel,
                })

    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows_for_csv)

    metrics = {
        "total_steps": total_steps,
        "hit_change_rate": hit_change / max(total_steps, 1),
        "hit_turn_rate": hit_turn / max(total_steps, 1),
        "hit_prestop_rate": hit_prestop / max(total_steps, 1),
        "change_window": args.change_window,
        "turn_window": args.turn_window,
        "prestop_window": args.prestop_window,
    }

    metrics_path = output_dir / "metrics.json"
    json.dump(metrics, open(metrics_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n[Eval done]")
    print("summary csv:", csv_path)
    print("summary jsonl:", jsonl_path)
    print("metrics:", metrics_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
