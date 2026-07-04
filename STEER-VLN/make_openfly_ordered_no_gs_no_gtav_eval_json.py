#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenFly STEER-VLN eval-json normalizer.

Purpose
-------
1) remove unavailable scenes by default:
   - env_gs_*
   - GTA / GTAV / GTA5 related scenes, e.g. env_game_gtav
2) reorder remaining samples so eval launches environments in this order:
   - env_airsim_26 first
   - other env_airsim_* scenes next
   - env_ue_* scenes last
3) keep the original sample order inside each environment.

The script intentionally writes both old and new output names:
  seen_no_gs_no_gtav.json
  seen_ordered_no_gs_no_gtav.json
so older run scripts and ordered run scripts stay compatible.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

DEFAULT_AIRSIM_ORDER = [
    "env_airsim_26",
    "env_airsim_16",
    "env_airsim_18",
    "env_airsim_23",
    "env_airsim_gz",
    "env_airsim_sh",
]


def env_name(item: dict) -> str:
    return str(item.get("image_path", "")).replace("\\", "/").split("/")[0]


def is_gs_env(env: str) -> bool:
    return str(env).lower().startswith("env_gs")


def is_gtav_env(env: str) -> bool:
    e = str(env).lower()
    return (
        "gtav" in e
        or "gta_v" in e
        or "gta-v" in e
        or "gta5" in e
        or e in {"env_game_gtav", "env_game_gta", "env_gtav"}
        or e.startswith("env_game_gtav")
    )


def should_remove(env: str, *, filter_gs: bool = True, filter_gtav: bool = True) -> bool:
    return (filter_gs and is_gs_env(env)) or (filter_gtav and is_gtav_env(env))


def env_rank(env: str, custom_order: List[str]) -> Tuple[int, int, str]:
    env = str(env)
    if env in custom_order:
        return (0, custom_order.index(env), env)
    if env.startswith("env_airsim_"):
        return (1, 0, env)
    if env.startswith("env_ue_"):
        return (2, 0, env)
    if is_gs_env(env):
        return (3, 0, env)
    if is_gtav_env(env):
        return (4, 0, env)
    return (5, 0, env)


def filter_and_order(data: list, custom_order: List[str], *, filter_gs: bool, filter_gtav: bool):
    kept = []
    removed = []
    for i, item in enumerate(data):
        env = env_name(item)
        if should_remove(env, filter_gs=filter_gs, filter_gtav=filter_gtav):
            removed.append((i, item))
        else:
            kept.append((i, item))

    ordered_pairs = sorted(
        kept,
        key=lambda pair: (*env_rank(env_name(pair[1]), custom_order), pair[0]),
    )
    return [x for _, x in ordered_pairs], [x for _, x in removed]


def first_env_order(data: Iterable[dict]) -> List[str]:
    order = []
    seen = set()
    for item in data:
        env = env_name(item)
        if env not in seen:
            order.append(env)
            seen.add(env)
    return order


def print_counts(title: str, data: list, custom_order: List[str]) -> None:
    c = Counter(env_name(x) for x in data)
    print(title)
    print(f"  total={len(data)}")
    for env, n in sorted(c.items(), key=lambda kv: (*env_rank(kv[0], custom_order), kv[0])):
        print(f"  {env}: {n}")


def write_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[WRITE] {path} ({len(data)} samples)")


def process_one(
    name: str,
    src: Path,
    out_dir: Path,
    custom_order: List[str],
    *,
    filter_gs: bool,
    filter_gtav: bool,
    write_legacy_alias: bool,
    rewrite_inplace: bool = False,
    backup: bool = True,
) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    kept, removed = filter_and_order(data, custom_order, filter_gs=filter_gs, filter_gtav=filter_gtav)

    print("=" * 88)
    print(f"[SRC] {src}")
    print_counts("[BEFORE]", data, custom_order)
    print_counts("[REMOVED]", removed, custom_order)
    print_counts("[AFTER ordered no_gs_no_gtav]", kept, custom_order)
    print("[ORDER]", " -> ".join(first_env_order(kept)) if kept else "<empty>")

    ordered_name = f"{name}_ordered_no_gs_no_gtav.json"
    write_json(out_dir / ordered_name, kept)

    if write_legacy_alias:
        legacy_name = f"{name}_no_gs_no_gtav.json"
        write_json(out_dir / legacy_name, kept)

    if rewrite_inplace:
        if backup:
            bak = src.with_suffix(src.suffix + ".bak")
            bak.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[BACKUP] {bak}")
        write_json(src, kept)

    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen", default="dataset/Annotation/seen.json")
    parser.add_argument("--unseen", default="dataset/Annotation/unseen.json")
    parser.add_argument("--eval_test", "--eval-test", default=None)
    parser.add_argument("--out_dir", "--out-dir", default="dataset/Annotation/filtered")
    parser.add_argument(
        "--only",
        choices=["all", "seen", "unseen", "eval_test"],
        default="all",
        help="Generate only one split. Existing run scripts use --only seen/unseen.",
    )
    parser.add_argument(
        "--airsim_order",
        default=",".join(DEFAULT_AIRSIM_ORDER),
        help="Comma-separated preferred AirSim env order. Default puts env_airsim_26 first.",
    )
    parser.add_argument("--keep_gs", action="store_true", help="Do not filter env_gs_* samples.")
    parser.add_argument("--keep_gtav", action="store_true", help="Do not filter GTA/GTAV samples.")
    parser.add_argument(
        "--no_legacy_alias",
        action="store_true",
        help="Only write *_ordered_no_gs_no_gtav.json, not legacy *_no_gs_no_gtav.json aliases.",
    )
    parser.add_argument(
        "--rewrite_eval_test",
        action="store_true",
        help="Also rewrite --eval_test in place after backing it up. Use for configs/eval_test.json.",
    )
    parser.add_argument("--no_backup", action="store_true")
    args = parser.parse_args()

    custom_order = [x.strip() for x in args.airsim_order.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    filter_gs = not args.keep_gs
    filter_gtav = not args.keep_gtav
    write_legacy_alias = not args.no_legacy_alias

    jobs = []
    if args.only in {"all", "seen"}:
        jobs.append(("seen", Path(args.seen), False))
    if args.only in {"all", "unseen"}:
        jobs.append(("unseen", Path(args.unseen), False))
    if args.eval_test and args.only in {"all", "eval_test"}:
        jobs.append(("eval_test", Path(args.eval_test), bool(args.rewrite_eval_test)))

    if not jobs:
        raise SystemExit("[FAIL] no job selected. Check --only and --eval_test.")

    for name, src, rewrite in jobs:
        if not src.is_file():
            raise SystemExit(f"[FAIL] missing input json: {src}")
        process_one(
            name,
            src,
            out_dir,
            custom_order,
            filter_gs=filter_gs,
            filter_gtav=filter_gtav,
            write_legacy_alias=write_legacy_alias,
            rewrite_inplace=rewrite,
            backup=not args.no_backup,
        )


if __name__ == "__main__":
    main()
