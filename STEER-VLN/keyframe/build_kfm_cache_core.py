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
import json
import re
from pathlib import Path
from typing import Dict, Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from keyframe.keyframe_scorer import AttentionKeyframeScorer, KeyframeScorerConfig
from keyframe.simple_tokenizer import SimpleTextTokenizer
from keyframe.parquet_keyframe_dataset import (
    ParquetKeyframeDataset,
    collate_keyframe_batch,
)


class ExcludePreviousCandidateWrapper:
    """
    Keep KFM candidate set consistent with final E5-S:
      keyframe != previous frame
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample = dict(sample)

        history_mask = sample["history_mask"].clone()
        soft_labels = sample["soft_labels"].clone()

        valid = torch.nonzero(history_mask.bool(), as_tuple=False).flatten()

        if valid.numel() >= 2:
            previous_local_idx = valid[-1]
            history_mask[previous_local_idx] = False
            soft_labels[previous_local_idx] = 0.0

            label_sum = soft_labels[history_mask.bool()].sum()
            if torch.isfinite(label_sum) and label_sum > 0:
                soft_labels = soft_labels / label_sum
            else:
                soft_labels.zero_()
                remain = torch.nonzero(history_mask.bool(), as_tuple=False).flatten()
                if remain.numel() > 0:
                    soft_labels[remain] = 1.0 / float(remain.numel())

            sample["history_mask"] = history_mask
            sample["soft_labels"] = soft_labels

            meta = dict(sample.get("meta", {}))
            meta["exclude_previous"] = True
            meta["previous_candidate_local_idx"] = int(previous_local_idx.item())
            sample["meta"] = meta

        return sample


def parquet_num_rows(path: str) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def safe_record_key(image_path: str, cur_idx: int) -> str:
    return f"{image_path}::{int(cur_idx)}"


def select_keyframe_from_logits(logits, probs, history_mask, candidate_indices):
    """
    Map model local top index [0, max_history) back to original frame index.
    candidate_indices is unpadded; valid candidates occupy the tail positions.
    """
    k = int(logits.numel())
    m = len(candidate_indices)
    offset = k - m

    masked_logits = logits.masked_fill(~history_mask.bool(), -1e4)
    top_local = int(masked_logits.argmax().item())

    cand_pos = top_local - offset
    if cand_pos < 0 or cand_pos >= m:
        return None, top_local, 0.0

    key_idx = int(candidate_indices[cand_pos])
    score = float(probs[top_local].item())
    return key_idx, top_local, score


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--vocab_path", type=str, default="")
    p.add_argument("--annotation_path", type=str, required=True)
    p.add_argument("--parquet_root", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)

    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--max_history", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min_timestep", type=int, default=2)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--cache_size", type=int, default=64)

    p.add_argument("--exclude_previous", action="store_true")
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    vocab_path = Path(args.vocab_path) if args.vocab_path else ckpt_path.parent / "simple_tokenizer_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab: {vocab_path}")

    tokenizer = SimpleTextTokenizer.load(vocab_path)

    cfg_dict = dict(ckpt["cfg"])
    cfg_dict["vocab_size"] = len(tokenizer)
    cfg_dict["max_history"] = args.max_history
    cfg_dict["image_size"] = args.image_size
    cfg = KeyframeScorerConfig(**cfg_dict)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    model = AttentionKeyframeScorer(cfg, padding_idx=tokenizer.pad_token_id)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    dataset = ParquetKeyframeDataset(
        annotation_path=args.annotation_path,
        parquet_root=args.parquet_root,
        tokenizer=tokenizer,
        max_history=args.max_history,
        image_size=args.image_size,
        max_episodes=None if args.max_episodes <= 0 else args.max_episodes,
        stride=args.stride,
        min_timestep=args.min_timestep,
        cache_size=args.cache_size,
        use_landmark_score=False,
    )

    if args.exclude_previous:
        dataset = ExcludePreviousCandidateWrapper(dataset)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=lambda b: collate_keyframe_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    row_count_cache: Dict[str, int] = {}
    records: Dict[str, Any] = {}

    total_seen = 0
    total_written = 0
    fallback_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="build learned keyframe cache"):
            history_images = batch["history_images"].to(device, non_blocking=True)
            current_image = batch["current_image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            motion_feats = batch["motion_feats"].to(device, non_blocking=True)
            history_mask = batch["history_mask"].to(device, non_blocking=True)

            logits = model(
                history_images=history_images,
                current_image=current_image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                motion_feats=motion_feats,
                history_mask=history_mask,
            )

            probs = torch.softmax(logits.masked_fill(~history_mask.bool(), -1e4), dim=-1)

            for i, meta in enumerate(batch["meta"]):
                total_seen += 1

                parquet_path = str(meta["parquet_path"])
                cur_idx = int(meta["current_idx"])
                image_path = str(meta.get("image_path", ""))

                if parquet_path not in row_count_cache:
                    row_count_cache[parquet_path] = parquet_num_rows(parquet_path)

                n = row_count_cache[parquet_path]

                # Match WaypointProbeDataset / OpenFlyFeatureDualDataset valid range:
                # t >= 2 and t + horizon < n
                if cur_idx < args.min_timestep:
                    continue
                if cur_idx + args.horizon >= n:
                    continue

                candidate_indices = list(meta["candidate_indices"])

                key_idx, top_local, score = select_keyframe_from_logits(
                    logits=logits[i].detach().cpu(),
                    probs=probs[i].detach().cpu(),
                    history_mask=history_mask[i].detach().cpu(),
                    candidate_indices=candidate_indices,
                )

                prev_idx = max(0, cur_idx - 1)

                if key_idx is None:
                    key_idx = max(0, cur_idx - 2)
                    fallback_count += 1

                if args.exclude_previous and key_idx == prev_idx and cur_idx >= 2:
                    key_idx = max(0, cur_idx - 2)
                    fallback_count += 1

                if key_idx >= cur_idx:
                    key_idx = max(0, cur_idx - 2)
                    fallback_count += 1

                record_candidate_indices = list(candidate_indices)

                if args.exclude_previous:
                    record_candidate_indices = [
                        int(x) for x in record_candidate_indices
                        if int(x) != int(prev_idx)
                    ]

                record = {
                    "image_path": image_path,
                    "parquet_path": parquet_path,
                    "current_idx": cur_idx,
                    "previous_idx": prev_idx,
                    "keyframe_idx": int(key_idx),
                    "score": float(score),
                    "top_local_idx": int(top_local),
                    "candidate_indices": record_candidate_indices,
                    "exclude_previous": bool(args.exclude_previous),
                }

                records[safe_record_key(image_path, cur_idx)] = record
                total_written += 1

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"[OK] saved cache: {out}")
    print(f"seen={total_seen}, written={total_written}, fallback={fallback_count}")


if __name__ == "__main__":
    main()
