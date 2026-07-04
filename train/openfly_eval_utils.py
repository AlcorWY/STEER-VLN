"""
OpenFly evaluation split and metrics utilities.
Aligned for STEER_VLN experiments.

This file is intentionally kept under train/ so the original OpenFly source
entrypoints can use the same split names and source-compatible metrics as STEER_VLN.
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _norm_split(x: Optional[str]) -> str:
    x = (x or "demo").strip().lower().replace("-", "_")
    aliases = {
        "eval": "demo",
        "eval_test": "demo",
        "test": "demo",
        "seen": "test_seen",
        "testseen": "test_seen",
        "test_seen": "test_seen",
        "test_unseen": "test_unseen",
        "testunseen": "test_unseen",
        "unseen": "test_unseen",
        "custom": "custom",
        "demo": "demo",
    }
    return aliases.get(x, x)


def _existing(paths: Sequence[str]) -> Optional[str]:
    for p in paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = PROJECT_ROOT / pp
        if pp.is_file():
            return str(pp)
    return None


def resolve_eval_config(split: Optional[str] = None, explicit_config: Optional[str] = None) -> str:
    """Resolve EVAL_CONFIG consistently for OpenFly train/ and STEER_VLN.

    Priority:
      1. explicit_config argument
      2. EVAL_CONFIG environment variable
      3. split-specific environment variables
      4. common file names under configs/ and dataset/Annotation/
    """
    if explicit_config:
        p = Path(explicit_config)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"explicit eval config not found: {p}")

    if os.environ.get("EVAL_CONFIG"):
        p = Path(os.environ["EVAL_CONFIG"])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"EVAL_CONFIG not found: {p}")

    split = _norm_split(split or os.environ.get("OPENFLY_EVAL_SPLIT") or os.environ.get("STEER_VLN_EVAL_SPLIT") or "demo")

    if split == "custom":
        raise FileNotFoundError("split=custom requires EVAL_CONFIG=/path/to/eval.json")

    if split == "demo":
        candidates = [
            "configs/eval_test.json",
            "dataset/Annotation/eval_test.json",
        ]
    elif split == "test_seen":
        if os.environ.get("OPENFLY_TEST_SEEN_CONFIG"):
            return resolve_eval_config(explicit_config=os.environ["OPENFLY_TEST_SEEN_CONFIG"])
        if os.environ.get("STEER_VLN_TEST_SEEN_CONFIG"):
            return resolve_eval_config(explicit_config=os.environ["STEER_VLN_TEST_SEEN_CONFIG"])
        candidates = [
            "configs/eval_test_seen.json",
            "configs/eval_seen.json",
            "configs/test_seen.json",
            "configs/test-seen.json",
            "configs/seen.json",
            "dataset/Annotation/eval_test_seen.json",
            "dataset/Annotation/eval_seen.json",
            "dataset/Annotation/test_seen.json",
            "dataset/Annotation/test-seen.json",
            "dataset/Annotation/seen.json",
            "Annotation/seen.json",
            "dataset/Annotation/test_seen_airsim16.json",
            "dataset/Annotation/test-seen_airsim16.json",
        ]
    elif split == "test_unseen":
        if os.environ.get("OPENFLY_TEST_UNSEEN_CONFIG"):
            return resolve_eval_config(explicit_config=os.environ["OPENFLY_TEST_UNSEEN_CONFIG"])
        if os.environ.get("STEER_VLN_TEST_UNSEEN_CONFIG"):
            return resolve_eval_config(explicit_config=os.environ["STEER_VLN_TEST_UNSEEN_CONFIG"])
        candidates = [
            "configs/eval_test_unseen.json",
            "configs/eval_unseen.json",
            "configs/test_unseen.json",
            "configs/test-unseen.json",
            "configs/unseen.json",
            "dataset/Annotation/eval_test_unseen.json",
            "dataset/Annotation/eval_unseen.json",
            "dataset/Annotation/test_unseen.json",
            "dataset/Annotation/test-unseen.json",
            "dataset/Annotation/unseen.json",
            "Annotation/unseen.json",
            "dataset/Annotation/test_unseen_airsim16.json",
            "dataset/Annotation/test-unseen_airsim16.json",
        ]
    else:
        raise ValueError(f"unknown eval split: {split}; use demo, test_seen, test_unseen, or custom")

    found = _existing(candidates)
    if found is None:
        msg = [
            f"Could not resolve eval config for split={split}.",
            "Set one of:",
            "  EVAL_CONFIG=/path/to/eval.json",
            "  OPENFLY_TEST_SEEN_CONFIG=/path/to/seen.json",
            "  OPENFLY_TEST_UNSEEN_CONFIG=/path/to/unseen.json",
            "  STEER_VLN_TEST_SEEN_CONFIG=/path/to/seen.json",
            "  STEER_VLN_TEST_UNSEEN_CONFIG=/path/to/unseen.json",
            "Checked candidates:",
        ]
        msg.extend([f"  - {c}" for c in candidates])
        raise FileNotFoundError("\n".join(msg))
    return found


def get_eval_split_name() -> str:
    return _norm_split(os.environ.get("OPENFLY_EVAL_SPLIT") or os.environ.get("STEER_VLN_EVAL_SPLIT") or os.environ.get("STEER_VLN_EVAL_SPLIT_NAME") or "demo")


def load_eval_info(split: Optional[str] = None, explicit_config: Optional[str] = None):
    cfg = resolve_eval_config(split=split, explicit_config=explicit_config)
    with open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, cfg, get_eval_split_name() if split is None else _norm_split(split)


class OpenFlyMetricAccumulator:
    """Source-compatible NE/SR/OSR/SPL accumulator.

    NE/m: mean final distance-to-goal.
    SR/%: mean success * 100 with source threshold dis < 20.
    OSR/%: mean oracle success * 100, if trajectory ever reached dis < 20.
    SPL/%: mean source-compatible OpenFly SPL * 100. When both formulas
           are available, this uses traj_len / pass_len when success else 0.
    """

    def __init__(self):
        self.distance_to_goal: List[float] = []
        self.success: List[float] = []
        self.osr: List[float] = []
        self.spl: List[float] = []

    def append(self, ne: float, success: float, osr: float, spl: float):
        self.distance_to_goal.append(float(ne))
        self.success.append(float(success))
        self.osr.append(float(osr))
        self.spl.append(float(spl))

    def update_from_bridge_latest(self, bridge):
        # Prefer OpenFly source-compatible SPL when both formulas are available.
        spl_value = bridge.spl_raw[-1] if hasattr(bridge, "spl_raw") and bridge.spl_raw else bridge.spl[-1]
        self.append(
            bridge.distance_to_goal[-1],
            bridge.success[-1],
            bridge.osr[-1],
            spl_value,
        )

    def summary(self) -> Dict[str, float]:
        n = len(self.distance_to_goal)
        if n <= 0:
            return {"NE/m": 0.0, "SR/%": 0.0, "OSR/%": 0.0, "SPL/%": 0.0, "count": 0}
        return {
            "NE/m": sum(self.distance_to_goal) / n,
            "SR/%": 100.0 * sum(self.success) / n,
            "OSR/%": 100.0 * sum(self.osr) / n,
            "SPL/%": 100.0 * sum(self.spl) / n,
            "count": n,
        }

    def print_summary(self, prefix: str = "OpenFly Metrics"):
        s = self.summary()
        print(f"NE/m: {s['NE/m']:.4f}")
        print(f"SR/%: {s['SR/%']:.2f}")
        print(f"OSR/%: {s['OSR/%']:.2f}")
        print(f"SPL/%: {s['SPL/%']:.2f}")
        print(
            f"{prefix}: NE/m={s['NE/m']:.4f}, "
            f"SR/%={s['SR/%']:.2f}, OSR/%={s['OSR/%']:.2f}, SPL/%={s['SPL/%']:.2f}"
        )
        return s
