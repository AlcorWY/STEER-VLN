#!/usr/bin/env python3
"""
Patch OpenFly/STEER-VLN eval scripts to report both OpenFly raw SPL and bounded standard SPL.

Targets by default, relative to project root:
  - STEER-VLN/eval_hd_lora.py
  - train/eval.py
  - train/eval_gtav.py
  - train/eval_baseline_stop_metrics.py, if it contains the same eval loop

Outputs added during eval:
  - per episode: SR, OSR, NE, SPL, SPL_raw, SPL_bounded
  - final summary: NE/m, SR/%, OSR/%, SPL/%, SPL_raw/%, SPL_bounded/%
  - scene-wise summary lines and CSV/JSON files under EVAL_METRICS_ROOT, default runs/eval_metrics
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HELPERS = r'''

# =========================
# OpenFly paper metric helpers
# =========================
def openfly_scene_category(item, env_name="unknown"):
    """
    Extract a scene/category label for scene-wise reporting.
    Typical image_path: env_airsim_16/astar_data/medium_average/xxx
    -> scene_category: medium_average
    """
    try:
        image_path = str(item.get("image_path", ""))
    except Exception:
        image_path = ""

    parts = image_path.replace("\\", "/").split("/")
    if "astar_data" in parts:
        i = parts.index("astar_data")
        if i + 1 < len(parts) and parts[i + 1]:
            return parts[i + 1]

    # fallback: use parent folder before file name
    if len(parts) >= 2 and parts[-2]:
        return parts[-2]

    return str(env_name)


def openfly_compute_spl_values(success, traj_len, pass_len):
    """
    SPL_raw follows OpenFly source: success * traj_len / pass_len.
    SPL_bounded follows standard reporting: success * traj_len / max(traj_len, pass_len).
    SPL/% in final summary uses SPL_bounded/%.
    """
    ref_len = max(float(traj_len), 1e-6)
    pred_len = max(float(pass_len), 1e-6)
    if int(success) == 1:
        spl_raw = ref_len / pred_len
        spl_bounded = ref_len / max(ref_len, pred_len)
        spl_bounded = max(0.0, min(1.0, float(spl_bounded)))
    else:
        spl_raw = 0.0
        spl_bounded = 0.0
    return float(spl_raw), float(spl_bounded)


def openfly_append_scene_metric(nav_episode_metrics, env_name, scene_category, sample_idx, ne_m, sr, osr, spl_raw, spl_bounded):
    nav_episode_metrics.append({
        "env": str(env_name),
        "scene": str(scene_category),
        "sample_idx": int(sample_idx),
        "NE/m": float(ne_m),
        "SR": int(sr),
        "OSR": int(osr),
        "SPL_raw": float(spl_raw),
        "SPL_bounded": float(spl_bounded),
        "SPL": float(spl_bounded),
    })


def openfly_print_and_save_metric_summary(nav_episode_metrics, metrics_tag="eval"):
    import csv as _csv
    import json as _json
    import os as _os
    from pathlib import Path as _Path
    from collections import defaultdict as _defaultdict

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def _summ(rows):
        return {
            "samples": len(rows),
            "NE/m": _mean([r["NE/m"] for r in rows]),
            "SR/%": 100.0 * _mean([r["SR"] for r in rows]),
            "OSR/%": 100.0 * _mean([r["OSR"] for r in rows]),
            "SPL/%": 100.0 * _mean([r["SPL_bounded"] for r in rows]),
            "SPL_raw/%": 100.0 * _mean([r["SPL_raw"] for r in rows]),
            "SPL_bounded/%": 100.0 * _mean([r["SPL_bounded"] for r in rows]),
        }

    total = _summ(nav_episode_metrics)

    print(f"NE/m: {total['NE/m']:.4f}")
    print(f"SR/%: {total['SR/%']:.2f}")
    print(f"OSR/%: {total['OSR/%']:.2f}")
    print(f"SPL/%: {total['SPL/%']:.2f}")
    print(f"SPL_raw/%: {total['SPL_raw/%']:.2f}")
    print(f"SPL_bounded/%: {total['SPL_bounded/%']:.2f}")

    by_scene = _defaultdict(list)
    by_env_scene = _defaultdict(list)
    for r in nav_episode_metrics:
        by_scene[r["scene"]].append(r)
        by_env_scene[(r["env"], r["scene"])].append(r)

    print("\nScene-wise metrics:")
    scene_rows = []
    for scene in sorted(by_scene.keys()):
        info = _summ(by_scene[scene])
        row = {"scene": scene, **info}
        scene_rows.append(row)
        print(
            f"[SceneMetrics] scene={scene} samples={info['samples']} "
            f"NE/m={info['NE/m']:.4f} SR/%={info['SR/%']:.2f} OSR/%={info['OSR/%']:.2f} "
            f"SPL/%={info['SPL/%']:.2f} SPL_raw/%={info['SPL_raw/%']:.2f} "
            f"SPL_bounded/%={info['SPL_bounded/%']:.2f}"
        )

    print("\nEnv-scene-wise metrics:")
    env_scene_rows = []
    for env, scene in sorted(by_env_scene.keys()):
        info = _summ(by_env_scene[(env, scene)])
        row = {"env": env, "scene": scene, **info}
        env_scene_rows.append(row)
        print(
            f"[EnvSceneMetrics] env={env} scene={scene} samples={info['samples']} "
            f"NE/m={info['NE/m']:.4f} SR/%={info['SR/%']:.2f} OSR/%={info['OSR/%']:.2f} "
            f"SPL/%={info['SPL/%']:.2f} SPL_raw/%={info['SPL_raw/%']:.2f} "
            f"SPL_bounded/%={info['SPL_bounded/%']:.2f}"
        )

    metrics_root = _Path(_os.environ.get("EVAL_METRICS_ROOT", "runs/eval_metrics"))
    metrics_root.mkdir(parents=True, exist_ok=True)
    safe_tag = str(metrics_tag).replace("/", "_").replace("\\", "_").replace(" ", "_")

    episode_csv = metrics_root / f"{safe_tag}_episode_metrics.csv"
    scene_csv = metrics_root / f"{safe_tag}_scene_metrics.csv"
    env_scene_csv = metrics_root / f"{safe_tag}_env_scene_metrics.csv"
    summary_json = metrics_root / f"{safe_tag}_metric_summary.json"

    episode_fields = ["env", "scene", "sample_idx", "NE/m", "SR", "OSR", "SPL/%", "SPL_raw/%", "SPL_bounded/%", "SPL", "SPL_raw", "SPL_bounded"]
    with open(episode_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=episode_fields)
        w.writeheader()
        for r in nav_episode_metrics:
            rr = dict(r)
            rr["SPL/%"] = 100.0 * rr["SPL_bounded"]
            rr["SPL_raw/%"] = 100.0 * rr["SPL_raw"]
            rr["SPL_bounded/%"] = 100.0 * rr["SPL_bounded"]
            w.writerow({k: rr.get(k, "") for k in episode_fields})

    scene_fields = ["scene", "samples", "NE/m", "SR/%", "OSR/%", "SPL/%", "SPL_raw/%", "SPL_bounded/%"]
    with open(scene_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=scene_fields)
        w.writeheader()
        for r in scene_rows:
            w.writerow({k: r.get(k, "") for k in scene_fields})

    env_scene_fields = ["env", "scene", "samples", "NE/m", "SR/%", "OSR/%", "SPL/%", "SPL_raw/%", "SPL_bounded/%"]
    with open(env_scene_csv, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=env_scene_fields)
        w.writeheader()
        for r in env_scene_rows:
            w.writerow({k: r.get(k, "") for k in env_scene_fields})

    with open(summary_json, "w", encoding="utf-8") as f:
        _json.dump({
            "total": total,
            "scene_metrics": scene_rows,
            "env_scene_metrics": env_scene_rows,
            "episode_metrics_csv": str(episode_csv),
            "scene_metrics_csv": str(scene_csv),
            "env_scene_metrics_csv": str(env_scene_csv),
        }, f, ensure_ascii=False, indent=2)

    print(f"[MetricFiles] episode_metrics_csv={episode_csv}")
    print(f"[MetricFiles] scene_metrics_csv={scene_csv}")
    print(f"[MetricFiles] env_scene_metrics_csv={env_scene_csv}")
    print(f"[MetricFiles] summary_json={summary_json}")

    return total, scene_rows, env_scene_rows
# =========================
# End OpenFly paper metric helpers
# =========================
'''


def add_helpers(s: str) -> str:
    if "def openfly_compute_spl_values" in s:
        return s
    marker = "from common import *\n"
    if marker in s:
        return s.replace(marker, marker + HELPERS + "\n", 1)
    # fallback: prepend after first imports block
    return HELPERS + "\n" + s


def patch_spl_lists(s: str) -> str:
    if "self.spl_raw = []" not in s:
        s = s.replace("self.spl = []", "self.spl = []\n        self.spl_raw = []\n        self.spl_bounded = []")
    return s


def patch_print_info(s: str) -> str:
    old_print = '        print(f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}")\n'
    old_return = '        return f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, SPL: {self.spl[-1]}"\n'
    new = '''        _spl_raw_list = getattr(self, "spl_raw", self.spl)
        _spl_bounded_list = getattr(self, "spl_bounded", self.spl)
        _spl_raw = _spl_raw_list[-1] if _spl_raw_list else self.spl[-1]
        _spl_bounded = _spl_bounded_list[-1] if _spl_bounded_list else self.spl[-1]
        print(
            f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, "
            f"SPL: {_spl_bounded}, SPL_raw: {_spl_raw}, SPL_bounded: {_spl_bounded}"
        )
'''
    new_return = '''        return (
            f"SR: {self.success[-1]}, OSR: {self.osr[-1]}, NE: {self.distance_to_goal[-1]}, "
            f"SPL: {_spl_bounded}, SPL_raw: {_spl_raw}, SPL_bounded: {_spl_bounded}"
        )
'''
    s = s.replace(old_print, new)
    s = s.replace(old_return, new_return)
    return s


def patch_nav_list_init(s: str) -> str:
    if "nav_episode_metrics = []" in s:
        return s
    pattern = re.compile(r"(\n(?P<indent>\s*)acc\s*=\s*0\n(?P=indent)data_num\s*=\s*0\n)")
    def repl(m):
        indent = m.group("indent")
        return m.group(1) + f"{indent}# Episode-level paper metrics for final and scene-wise reporting.\n{indent}nav_episode_metrics = []\n"
    s2, n = pattern.subn(repl, s, count=1)
    return s2


def patch_episode_spl_block(s: str) -> str:
    if "sample_spl_raw, sample_spl_bounded = openfly_compute_spl_values" in s:
        return s

    old = '''            if dis < 20:
                env_bridge.success.append(1)
                env_bridge.spl.append(env_bridge.traj_len / env_bridge.pass_len)
                acc += 1
            else:
                env_bridge.success.append(0)
                env_bridge.spl.append(0)
'''
    new = '''            sample_success = 1 if dis < 20 else 0
            sample_spl_raw, sample_spl_bounded = openfly_compute_spl_values(
                sample_success,
                env_bridge.traj_len,
                env_bridge.pass_len,
            )

            if sample_success:
                env_bridge.success.append(1)
                # Keep env_bridge.spl as paper-standard bounded SPL.
                env_bridge.spl.append(sample_spl_bounded)
                if not hasattr(env_bridge, "spl_raw"):
                    env_bridge.spl_raw = []
                if not hasattr(env_bridge, "spl_bounded"):
                    env_bridge.spl_bounded = []
                env_bridge.spl_raw.append(sample_spl_raw)
                env_bridge.spl_bounded.append(sample_spl_bounded)
                acc += 1
            else:
                env_bridge.success.append(0)
                env_bridge.spl.append(0.0)
                if not hasattr(env_bridge, "spl_raw"):
                    env_bridge.spl_raw = []
                if not hasattr(env_bridge, "spl_bounded"):
                    env_bridge.spl_bounded = []
                env_bridge.spl_raw.append(0.0)
                env_bridge.spl_bounded.append(0.0)
'''
    if old in s:
        return s.replace(old, new, 1)

    # More flexible regex for minor variants.
    pattern = re.compile(
        r"(\n\s*if\s+dis\s*<\s*20\s*:\s*\n"
        r"\s*env_bridge\.success\.append\(1\)\s*\n"
        r"\s*env_bridge\.spl\.append\(env_bridge\.traj_len\s*/\s*env_bridge\.pass_len\)\s*\n"
        r"\s*acc\s*\+=\s*1\s*\n"
        r"\s*else\s*:\s*\n"
        r"\s*env_bridge\.success\.append\(0\)\s*\n"
        r"\s*env_bridge\.spl\.append\(0\.?0?\)\s*\n)"
    )
    s2, n = pattern.subn("\n" + new, s, count=1)
    return s2


def patch_episode_metric_append(s: str) -> str:
    if "openfly_append_scene_metric(\n                nav_episode_metrics" in s:
        return s
    old = '''            if flag_osr == 0:
                env_bridge.osr.append(0)

            env_bridge.print_info()
'''
    new = '''            if flag_osr == 0:
                env_bridge.osr.append(0)

            sample_scene_category = openfly_scene_category(item, env_name)
            openfly_append_scene_metric(
                nav_episode_metrics,
                env_name=env_name,
                scene_category=sample_scene_category,
                sample_idx=idx,
                ne_m=float(env_bridge.distance_to_goal[-1]),
                sr=int(env_bridge.success[-1]),
                osr=int(env_bridge.osr[-1]),
                spl_raw=float(env_bridge.spl_raw[-1]),
                spl_bounded=float(env_bridge.spl_bounded[-1]),
            )

            env_bridge.print_info()
            print(
                f"[EpisodeMetric] env={env_name} scene={sample_scene_category} sample={idx} "
                f"NE/m={float(env_bridge.distance_to_goal[-1]):.4f} "
                f"SR={int(env_bridge.success[-1])} OSR={int(env_bridge.osr[-1])} "
                f"SPL={float(env_bridge.spl_bounded[-1]):.6f} "
                f"SPL_raw={float(env_bridge.spl_raw[-1]):.6f} "
                f"SPL_bounded={float(env_bridge.spl_bounded[-1]):.6f}"
            )
'''
    if old in s:
        return s.replace(old, new, 1)
    # fallback if no blank line exact
    marker = "            env_bridge.print_info()\n"
    insert = new
    if marker in s:
        return s.replace(marker, insert, 1)
    return s


def patch_save_eval_summary(s: str) -> str:
    if '"SPL_raw"' in s and 'additional_info={' in s:
        return s
    old = '''                stop_pred=sample_pred_stop,
                image_error=image_error,
            )
'''
    new = '''                stop_pred=sample_pred_stop,
                image_error=image_error,
                additional_info={
                    "scene_category": sample_scene_category,
                    "NE/m": float(env_bridge.distance_to_goal[-1]),
                    "SR": int(env_bridge.success[-1]),
                    "OSR": int(env_bridge.osr[-1]),
                    "SPL": float(env_bridge.spl_bounded[-1]),
                    "SPL_raw": float(env_bridge.spl_raw[-1]),
                    "SPL_bounded": float(env_bridge.spl_bounded[-1]),
                    "SR/%": 100.0 * float(env_bridge.success[-1]),
                    "OSR/%": 100.0 * float(env_bridge.osr[-1]),
                    "SPL/%": 100.0 * float(env_bridge.spl_bounded[-1]),
                    "SPL_raw/%": 100.0 * float(env_bridge.spl_raw[-1]),
                    "SPL_bounded/%": 100.0 * float(env_bridge.spl_bounded[-1]),
                },
            )
'''
    if old in s:
        return s.replace(old, new, 1)
    return s


def patch_final_summary(s: str) -> str:
    if "openfly_print_and_save_metric_summary(nav_episode_metrics, metrics_tag=metrics_tag)" in s:
        return s
    old = '''    print(f"Final accuracy: {final_acc:.4f}")
'''
    new = '''    print(f"Final accuracy: {final_acc:.4f}")

    metrics_tag = os.environ.get(
        "EVAL_METRICS_TAG",
        f"{Path(__file__).stem}_{baseline_method}" if "baseline_method" in globals() or "baseline_method" in locals() else Path(__file__).stem,
    )
    openfly_print_and_save_metric_summary(nav_episode_metrics, metrics_tag=metrics_tag)
'''
    if old in s:
        return s.replace(old, new, 1)
    return s


def patch_file(path: Path) -> bool:
    if not path.exists():
        print(f"[SKIP] missing {path}")
        return False
    s = path.read_text(encoding="utf-8", errors="ignore")
    orig = s
    if "env_bridge.spl.append(env_bridge.traj_len / env_bridge.pass_len)" not in s and "openfly_compute_spl_values" not in s:
        print(f"[SKIP] no OpenFly SPL pattern in {path}")
        return False

    s = add_helpers(s)
    s = patch_spl_lists(s)
    s = patch_print_info(s)
    s = patch_nav_list_init(s)
    s = patch_episode_spl_block(s)
    s = patch_episode_metric_append(s)
    s = patch_save_eval_summary(s)
    s = patch_final_summary(s)

    if s != orig:
        backup = path.with_suffix(path.suffix + ".spl_patch_bak")
        if not backup.exists():
            backup.write_text(orig, encoding="utf-8")
        path.write_text(s, encoding="utf-8")
        print(f"[OK] patched {path}")
        return True
    print(f"[OK] already patched {path}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="OpenFly-Platform project root")
    ap.add_argument("--targets", nargs="*", default=[
        "STEER-VLN/eval_hd_lora.py",
        "train/eval.py",
        "train/eval_gtav.py",
        "train/eval_baseline_stop_metrics.py",
    ])
    args = ap.parse_args()
    root = Path(args.root).resolve()
    for rel in args.targets:
        patch_file(root / rel)

if __name__ == "__main__":
    main()
