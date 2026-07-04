#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq


def wrap_angle(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def action_type_value_to_id(action_type, action_value):
    s = str(action_type).lower()
    try:
        v = float(action_value)
    except Exception:
        v = 0.0

    if "stop" in s:
        return 0

    if "forward" in s:
        if abs(v - 9) < 1e-3:
            return 9
        if abs(v - 6) < 1e-3:
            return 8
        return 1

    if "turn_left" in s:
        return 2
    if "turn_right" in s:
        return 3

    if "go_up" in s or s == "up" or "move_up" in s:
        return 4
    if "go_down" in s or s == "down" or "move_down" in s:
        return 5

    if "move_left" in s:
        return 6
    if "move_right" in s:
        return 7

    # fallback by value for some OpenFly variants
    if v == 2:
        return 2
    if v == 3:
        return 3

    return -1


def load_rows(parquet_path):
    rows = pq.read_table(parquet_path).to_pylist()
    return sorted(rows, key=lambda r: int(r["frame_index"]))


def waypoint_yaw(rows, i, horizon):
    j = i + horizon
    if j >= len(rows):
        return None

    p0 = np.asarray(rows[i]["pos"], dtype=np.float32)
    p1 = np.asarray(rows[j]["pos"], dtype=np.float32)
    yaw0 = float(rows[i]["yaw"])

    dx, dy, dz = (p1 - p0).tolist()
    if abs(dx) + abs(dy) < 1e-6:
        return 0.0

    target_yaw = math.atan2(dy, dx)
    return wrap_angle(target_yaw - yaw0)


def sign_name(x, eps=1e-3):
    if x > eps:
        return "positive"
    if x < -eps:
        return "negative"
    return "zero"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation_path", default="configs/eval_test.json")
    ap.add_argument("--parquet_root", default="dataset/hf_openfly_airsim16/traj")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--output_json", default="runs/STEER-VLN/offline_module_eval/left_right_yaw_consistency.json")
    args = ap.parse_args()

    ann = json.load(open(args.annotation_path, "r", encoding="utf-8"))
    if args.max_episodes > 0:
        ann = ann[:args.max_episodes]

    found = 0
    missing = 0

    action_counter = Counter()
    raw_type_counter = Counter()
    value_sign_by_action = defaultdict(Counter)
    next_yaw_sign_by_action = defaultdict(Counter)
    waypoint_sign_by_action = defaultdict(Counter)

    value_by_action = defaultdict(list)
    next_yaw_by_action = defaultdict(list)
    waypoint_yaw_by_action = defaultdict(list)

    turn_examples = []

    for item in ann:
        pp = Path(args.parquet_root) / Path(item.get("image_path", "")).with_suffix(".parquet")
        if not pp.exists():
            missing += 1
            continue

        found += 1
        rows = load_rows(pp)

        for i, r in enumerate(rows):
            action_type = r.get("action_type", "")
            action_value = r.get("action_value", 0.0)
            aid = action_type_value_to_id(action_type, action_value)

            raw_type_counter[str(action_type)] += 1
            action_counter[aid] += 1

            try:
                av = float(action_value)
            except Exception:
                av = 0.0

            value_by_action[aid].append(av)
            value_sign_by_action[aid][sign_name(av)] += 1

            if i + 1 < len(rows):
                dy = wrap_angle(float(rows[i + 1]["yaw"]) - float(rows[i]["yaw"]))
                next_yaw_by_action[aid].append(dy)
                next_yaw_sign_by_action[aid][sign_name(dy)] += 1

            wy = waypoint_yaw(rows, i, args.horizon)
            if wy is not None:
                waypoint_yaw_by_action[aid].append(wy)
                waypoint_sign_by_action[aid][sign_name(wy)] += 1

            if aid in [2, 3] and len(turn_examples) < 30:
                turn_examples.append({
                    "parquet": str(pp),
                    "frame_index": r.get("frame_index"),
                    "action_id": aid,
                    "action_type": str(action_type),
                    "action_value": av,
                    "value_sign": sign_name(av),
                    "next_yaw_deg": float(next_yaw_by_action[aid][-1] * 180 / math.pi) if next_yaw_by_action[aid] else None,
                    "waypoint_yaw_deg": float(wy * 180 / math.pi) if wy is not None else None,
                })

    def summarize_values(vals):
        if not vals:
            return None
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "count": int(len(arr)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "mean_deg_if_angle": float(arr.mean() * 180 / math.pi),
            "median_deg_if_angle": float(np.median(arr) * 180 / math.pi),
        }

    result = {
        "annotation_path": args.annotation_path,
        "parquet_root": args.parquet_root,
        "found_parquet_episodes": found,
        "missing_parquet_episodes": missing,
        "action_counter": {str(k): int(v) for k, v in action_counter.items()},
        "raw_action_type_counter": dict(raw_type_counter.most_common(100)),
        "by_action": {},
        "turn_examples": turn_examples,
    }

    for aid in sorted(action_counter.keys()):
        result["by_action"][str(aid)] = {
            "count": int(action_counter[aid]),
            "action_value": summarize_values(value_by_action[aid]),
            "action_value_sign": dict(value_sign_by_action[aid]),
            "next_yaw_delta": summarize_values(next_yaw_by_action[aid]),
            "next_yaw_sign": dict(next_yaw_sign_by_action[aid]),
            "waypoint_yaw": summarize_values(waypoint_yaw_by_action[aid]),
            "waypoint_yaw_sign": dict(waypoint_sign_by_action[aid]),
        }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("============================================================")
    print("[LEFT/RIGHT CONSISTENCY - ROBUST]")
    print("saved:", out)
    print("found_parquet_episodes:", found)
    print("missing_parquet_episodes:", missing)
    print("------------------------------------------------------------")
    print("[ACTION_COUNTER]", result["action_counter"])
    print("------------------------------------------------------------")

    for aid, name in [(2, "turn_left"), (3, "turn_right"), (6, "move_left"), (7, "move_right")]:
        print(f"action {aid} / {name}:")
        print(json.dumps(result["by_action"].get(str(aid)), ensure_ascii=False, indent=2))

    print("------------------------------------------------------------")
    print("[INTERPRET]")

    a2 = result["by_action"].get("2")
    a3 = result["by_action"].get("3")

    if not a2 or not a3:
        print("Cannot decide: action 2 or action 3 is missing in matched parquet subset.")
        print("Use raw_action_type_counter and turn_examples to inspect dataset labels.")
    else:
        a2_val = a2["action_value"]["mean"] if a2["action_value"] else None
        a3_val = a3["action_value"]["mean"] if a3["action_value"] else None
        if a2_val is not None and a3_val is not None:
            if a2_val > 0 and a3_val < 0:
                print("OK by action_value: action 2 has positive value, action 3 has negative value.")
            elif a2_val < 0 and a3_val > 0:
                print("REVERSED by action_value: action 2 negative, action 3 positive.")
            else:
                print("AMBIGUOUS by action_value: signs are not separated.")

    print("============================================================")


if __name__ == "__main__":
    main()
