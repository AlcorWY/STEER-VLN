# ===== STEER_VLN isolated import path =====
import sys as _steer_vln_sys
from pathlib import Path as _STEER_VLNPath

_STEER_VLN_FILE = _STEER_VLNPath(__file__).resolve()
_STEER_VLN_DIR = _STEER_VLN_FILE.parents[1]
_STEER_VLN_ROOT = _STEER_VLN_DIR.parents[0]

for _p in [str(_STEER_VLN_DIR), str(_STEER_VLN_ROOT / "code"), str(_STEER_VLN_ROOT)]:
    if _p not in _steer_vln_sys.path:
        _steer_vln_sys.path.insert(0, _p)
# ===== end STEER_VLN isolated import path =====

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from keyframe.simple_tokenizer import SimpleTextTokenizer
from keyframe.keyframe_scorer import (
    AttentionKeyframeScorer,
    KeyframeScorerConfig,
    keyframe_soft_ce_loss,
    keyframe_top1_accuracy,
)
from keyframe.parquet_keyframe_dataset import (
    ParquetKeyframeDataset,
    collate_keyframe_batch,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser("STEER_VLN KFM full core training")

    p.add_argument("--annotation_path", type=str, required=True)
    p.add_argument("--parquet_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--max_history", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min_timestep", type=int, default=1)
    p.add_argument("--label_temperature", type=float, default=0.7)
    p.add_argument("--use_landmark_score", action="store_true")

    p.add_argument("--max_vocab_size", type=int, default=20000)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--text_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--cache_size", type=int, default=64)

    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.001)
    p.add_argument("--save_interval", type=int, default=1000)

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def make_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    drop_last,
    collate_fn,
    prefetch_factor,
):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)


def save_checkpoint(path, model, cfg, args, global_step, epoch, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": int(global_step),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "args": vars(args),
    }

    if extra:
        payload.update(extra)

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


@torch.no_grad()
def run_val(model, loader, device):
    model.eval()
    losses = []
    accs = []

    for batch in tqdm(loader, desc="val", leave=False):
        batch = move_to_device(batch, device)

        logits = model(
            history_images=batch["history_images"],
            current_image=batch["current_image"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            motion_feats=batch["motion_feats"],
            history_mask=batch["history_mask"],
        )

        loss = keyframe_soft_ce_loss(
            logits=logits,
            soft_labels=batch["soft_labels"],
            history_mask=batch["history_mask"],
        )

        acc = keyframe_top1_accuracy(
            logits=logits,
            soft_labels=batch["soft_labels"],
            history_mask=batch["history_mask"],
        )

        losses.append(float(loss.item()))
        accs.append(float(acc))

    if not losses:
        return None, None

    return float(np.mean(losses)), float(np.mean(accs))


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = SimpleTextTokenizer.from_annotation(
        args.annotation_path,
        max_vocab_size=args.max_vocab_size,
        min_freq=1,
    )
    tokenizer.save(output_dir / "simple_tokenizer_vocab.json")

    print(f"[KFM] device={device}", flush=True)
    print(f"[KFM] vocab_size={len(tokenizer)}", flush=True)
    print(f"[KFM] output_dir={output_dir}", flush=True)

    dataset = ParquetKeyframeDataset(
        annotation_path=args.annotation_path,
        parquet_root=args.parquet_root,
        tokenizer=tokenizer,
        max_history=args.max_history,
        image_size=args.image_size,
        max_episodes=None if args.max_episodes <= 0 else args.max_episodes,
        stride=args.stride,
        label_temperature=args.label_temperature,
        min_timestep=args.min_timestep,
        cache_size=args.cache_size,
        use_landmark_score=args.use_landmark_score,
    )

    if len(dataset) <= 0:
        raise RuntimeError("No keyframe training samples found.")

    if args.dry_run:
        sample = dataset[0]
        print("[DRY RUN] sample keys:", list(sample.keys()))
        print("[DRY RUN] history_images:", sample["history_images"].shape)
        print("[DRY RUN] current_image:", sample["current_image"].shape)
        print("[DRY RUN] input_ids:", sample["input_ids"].shape)
        print("[DRY RUN] history_mask:", sample["history_mask"])
        print("[DRY RUN] soft_labels:", sample["soft_labels"])
        print("[DRY RUN] meta:", sample["meta"])
        return

    val_size = int(len(dataset) * args.val_ratio)
    if val_size <= 0 and len(dataset) >= 100:
        val_size = max(1, len(dataset) // 1000)

    train_size = len(dataset) - val_size

    if val_size > 0:
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_set = dataset
        val_set = None

    print(
        f"[KFM] dataset={len(dataset)}, train={len(train_set)}, val={0 if val_set is None else len(val_set)}",
        flush=True,
    )

    collate_fn = lambda batch: collate_keyframe_batch(
        batch,
        pad_token_id=tokenizer.pad_token_id,
    )

    pin_memory = device.type == "cuda"

    train_loader = make_loader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
        prefetch_factor=args.prefetch_factor,
    )

    val_loader = None
    if val_set is not None:
        val_loader = make_loader(
            dataset=val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_fn,
            prefetch_factor=args.prefetch_factor,
        )

    cfg = KeyframeScorerConfig(
        image_size=args.image_size,
        vocab_size=len(tokenizer),
        text_dim=args.text_dim,
        hidden_dim=args.hidden_dim,
        max_history=args.max_history,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_temporal_self_attention=True,
        use_motion_residual=True,
    )

    model = AttentionKeyframeScorer(
        cfg,
        padding_idx=tokenizer.pad_token_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    use_amp = bool(args.bf16 and device.type == "cuda")
    amp_dtype = torch.bfloat16

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")

        for batch in pbar:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(
                    history_images=batch["history_images"],
                    current_image=batch["current_image"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    motion_feats=batch["motion_feats"],
                    history_mask=batch["history_mask"],
                )

                loss = keyframe_soft_ce_loss(
                    logits=logits,
                    soft_labels=batch["soft_labels"],
                    history_mask=batch["history_mask"],
                )

            if not torch.isfinite(loss):
                print(f"[KFM][WARN] non-finite loss at step={global_step + 1}: {loss}", flush=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                acc = keyframe_top1_accuracy(
                    logits=logits,
                    soft_labels=batch["soft_labels"],
                    history_mask=batch["history_mask"],
                )

            global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{acc:.3f}",
                step=global_step,
            )

            if args.save_interval > 0 and global_step % args.save_interval == 0:
                save_checkpoint(
                    output_dir / f"keyframe_scorer_step_{global_step}.pt",
                    model=model,
                    cfg=cfg,
                    args=args,
                    global_step=global_step,
                    epoch=epoch,
                    extra={"train_loss": float(loss.item()), "train_acc": float(acc)},
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        val_loss, val_acc = None, None
        if val_loader is not None:
            val_loss, val_acc = run_val(model, val_loader, device)
            if val_loss is not None:
                print(
                    f"[epoch {epoch}] val_loss={val_loss:.4f}, val_acc={val_acc:.4f}",
                    flush=True,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        output_dir / "keyframe_scorer_best.pt",
                        model=model,
                        cfg=cfg,
                        args=args,
                        global_step=global_step,
                        epoch=epoch,
                        extra={"best_val_loss": best_val_loss, "val_acc": val_acc},
                    )

        save_checkpoint(
            output_dir / "keyframe_scorer_last.pt",
            model=model,
            cfg=cfg,
            args=args,
            global_step=global_step,
            epoch=epoch,
            extra={"val_loss": val_loss, "val_acc": val_acc},
        )

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if not (output_dir / "keyframe_scorer_best.pt").exists():
        save_checkpoint(
            output_dir / "keyframe_scorer_best.pt",
            model=model,
            cfg=cfg,
            args=args,
            global_step=global_step,
            epoch=max(args.epochs - 1, 0),
            extra={"fallback_best_from_last": True, "best_val_loss": best_val_loss},
        )

    print("[KFM] Training done.", flush=True)
    print("[KFM] best:", output_dir / "keyframe_scorer_best.pt", flush=True)
    print("[KFM] last:", output_dir / "keyframe_scorer_last.pt", flush=True)
    print("[KFM] vocab:", output_dir / "simple_tokenizer_vocab.json", flush=True)


if __name__ == "__main__":
    main()
