#!/usr/bin/env python3
"""Aggregate summary.jsonl files under images/eval_baseline/* and write a CSV summary."""
import csv
import json
from pathlib import Path

ROOT = Path("images") / "eval_baseline"
OUT_DIR = Path("runs/STEER-VLN/eval_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "summary_aggregate.csv"
OUT_STATS = OUT_DIR / "summary_stats.csv"

seen = set()
rows = []
for method_dir in sorted(ROOT.iterdir()):
    if not method_dir.is_dir():
        continue
    # collect candidate summary files: method-level and env-level
    candidates = []
    method_summary = method_dir / "summary.jsonl"
    if method_summary.exists():
        candidates.append(method_summary)
    for env_dir in sorted(method_dir.iterdir()):
        env_summary = env_dir / "summary.jsonl"
        if env_summary.exists():
            candidates.append(env_summary)

    for s in candidates:
        if str(s) in seen:
            continue
        seen.add(str(s))
        with open(s, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                key = (obj.get("baseline_method"), obj.get("env_name"), obj.get("sample_idx"))
                # avoid duplicate sample entries
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "baseline_method": obj.get("baseline_method", ""),
                    "env_name": obj.get("env_name", ""),
                    "sample_idx": obj.get("sample_idx", ""),
                    "instruction": obj.get("instruction", ""),
                    "success": int(obj.get("success", 0) or 0),
                    "osr": int(obj.get("osr", 0) or 0),
                    "num_steps": int(obj.get("num_steps", 0) or 0),
                    "predicted_stop": bool(obj.get("predicted_stop", False)),
                })

# write per-sample CSV
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["baseline_method", "env_name", "sample_idx", "instruction", "success", "osr", "num_steps", "predicted_stop"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# compute grouped stats
from collections import defaultdict

groups = defaultdict(list)
for r in rows:
    groups[(r["baseline_method"], r["env_name"])].append(r)

stats = []
for (method, env), items in sorted(groups.items()):
    n = len(items)
    if n == 0:
        continue
    sr = sum(i["success"] for i in items) / n
    avg_steps = sum(i["num_steps"] for i in items) / n
    stop_rate = sum(1 for i in items if i["predicted_stop"]) / n
    stats.append({
        "baseline_method": method,
        "env_name": env,
        "num_samples": n,
        "success_rate": round(sr, 4),
        "avg_num_steps": round(avg_steps, 2),
        "predicted_stop_rate": round(stop_rate, 4),
    })

with open(OUT_STATS, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["baseline_method", "env_name", "num_samples", "success_rate", "avg_num_steps", "predicted_stop_rate"])
    writer.writeheader()
    for s in stats:
        writer.writerow(s)

print(f"Wrote CSVs: {OUT_CSV}, {OUT_STATS}")
