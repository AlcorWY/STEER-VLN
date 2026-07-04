#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pyarrow.parquet as pq


def wrap_angle(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def load_rows(parquet_path):
    rows = pq.read_table(parquet_path).to_pylist()
    return sorted(rows, key=lambda r: int(r["frame_index"]))


def sign_name(x, eps=1e-4):
    if x > eps:
        return "positive"
    if x < -eps:
        return "negative"
    return "zero"


def deg(x):
    return x * 180.0 / math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation_path", default="configs/eval_test.json")
    ap.add_argument("--parquet_root", default="dataset/hf_openfly_airsim16/traj")
    ap.add_argument("--output_json", default="runs/STEER-VLN/offline_module_eval/turn_left_right_transition.json")
    args = ap.parse_args()

    ann = json.load(open(args.annotation_path, "r", encoding="utf-8"))
    root = Path(args.parquet_root)

    found = 0
    missing = 0

    raw_counter = Counter()
    stats = defaultdict(lambda: {
        "prev_to_current": [],
        "current_to_next": [],
        "examples": [],
    })

    for item in ann:
        pp = root / Path(item.get("image_path", "")).with_suffix(".parquet")
        if not pp.exists():
            missing += 1
            continue

        found += 1
        rows = load_rows(pp)

        for i, r in enumerate(rows):
            action_type = str(r.get("action_type", "")).strip().lower()
            raw_counter[action_type] += 1

            if action_type not in ["turn left", "turn right"]:
                continue

            yaw_cur = float(rows[i]["yaw"])

            prev_delta = None
            next_delta = None

            if i > 0:
                yaw_prev = float(rows[i - 1]["yaw"])
                prev_delta = wrap_angle(yaw_cur - yaw_prev)
                stats[action_type]["prev_to_current"].append(prev_delta)

            if i + 1 < len(rows):
                yaw_next = float(rows[i + 1]["yaw"])
                next_delta = wrap_angle(yaw_next - yaw_cur)
                stats[action_type]["current_to_next"].append(next_delta)

            if len(stats[action_type]["examples"]) < 20:
                ex = {
                    "parquet": str(pp),
                    "row_i": i,
                    "frame_index": r.get("frame_index"),
                    "action_type": action_type,
                    "action_value": r.get("action_value"),
                    "yaw_prev": float(rows[i - 1]["yaw"]) if i > 0 else None,
                    "yaw_cur": yaw_cur,
                    "yaw_next": float(rows[i + 1]["yaw"]) if i + 1 < len(rows) else None,
                    "prev_to_current_deg": deg(prev_delta) if prev_delta is not None else None,
                    "current_to_next_deg": deg(next_delta) if next_delta is not None else None,
                    "pos": r.get("pos"),
                }
                stats[action_type]["examples"].append(ex)

    result = {
        "annotation_path": args.annotation_path,
        "parquet_root": args.parquet_root,
        "found_parquet_episodes": found,
        "missing_parquet_episodes": missing,
        "raw_action_type_counter": dict(raw_counter.most_common(50)),
        "turn_stats": {},
    }

    print("============================================================")
    print("[TURN LEFT / RIGHT TRANSITION CHECK]")
    print("found_parquet_episodes:", found)
    print("missing_parquet_episodes:", missing)
    print("------------------------------------------------------------")
    print("[RAW_ACTION_TYPE_COUNTER]")
    for k, v in raw_counter.most_common(20):
        print(repr(k), v)

    for action_type in ["turn left", "turn right"]:
        result["turn_stats"][action_type] = {}

        print("------------------------------------------------------------")
        print(f"[{action_type}]")

        for key in ["prev_to_current", "current_to_next"]:
            vals = stats[action_type][key]
            arr = np.asarray(vals, dtype=np.float64)

            if len(arr) == 0:
                info = None
            else:
                signs = Counter(sign_name(float(x)) for x in arr)
                info = {
                    "count": int(len(arr)),
                    "mean_deg": float(deg(arr.mean())),
                    "median_deg": float(deg(np.median(arr))),
                    "positive": int(signs["positive"]),
                    "negative": int(signs["negative"]),
                    "zero": int(signs["zero"]),
                }

            result["turn_stats"][action_type][key] = info
            print(key, info)

        result["turn_stats"][action_type]["examples"] = stats[action_type]["examples"]

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("------------------------------------------------------------")
    print("[INTERPRETATION]")
    left_next = result["turn_stats"].get("turn left", {}).get("current_to_next")
    right_next = result["turn_stats"].get("turn right", {}).get("current_to_next")

    if left_next and right_next:
        lm = left_next["mean_deg"]
        rm = right_next["mean_deg"]

        print(f"turn left  current→next mean = {lm:.2f} deg")
        print(f"turn right current→next mean = {rm:.2f} deg")

        if lm > 0 and rm < 0:
            print("OK: turn left = positive yaw, turn right = negative yaw.")
            print("STEER_VLN 当前 left/right 逻辑不需要反转。")
        elif lm < 0 and rm > 0:
            print("REVERSED: turn left = negative yaw, turn right = positive yaw.")
            print("STEER_VLN 当前 left/right 逻辑需要反转。")
        else:
            print("AMBIGUOUS: left/right signs are not clearly separated.")
    else:
        print("Cannot decide because turn left or turn right samples are missing.")

    print("saved:", out)
    print("============================================================")


if __name__ == "__main__":
    main()
