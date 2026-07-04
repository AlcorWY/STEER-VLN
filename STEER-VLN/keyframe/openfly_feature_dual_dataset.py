import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from keyframe.waypoint_probe_dataset import (
    WaypointProbeDataset,
    normalize_image_obj,
)


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


def _bucket_distance(d: float) -> str:
    d = float(abs(d))
    if d <= 4.5:
        return "near"
    if d <= 8.0:
        return "middle"
    return "far"


def _bucket_yaw(yaw: float) -> str:
    yaw = float(yaw)
    if yaw > 0.35:
        return "left"
    if yaw < -0.35:
        return "right"
    return "straight"


def _bucket_z(z: float) -> str:
    z = float(z)
    if z > 1.0:
        return "up"
    if z < -1.0:
        return "down"
    return "stable"


class OpenFlyFeatureDualDataset(WaypointProbeDataset):
    """
    Final STEER-VLN dataset.

    Frame order:
      [keyframe, previous, current]

    The returned instruction is already trend-conditioned:
      original instruction + auxiliary navigation trend state

    This makes LoRA learn:
      image + language + trend state -> final OpenFly action
    """

    def __init__(
        self,
        *args,
        prestop_window: int = 3,
        learned_keyframe_cache: Optional[str] = None,
        **kwargs,
    ):
        requested_keyframe_mode = kwargs.get("keyframe_mode", "label")

        if requested_keyframe_mode == "learned_cache":
            kwargs["keyframe_mode"] = "label"

        super().__init__(*args, **kwargs)

        self.keyframe_mode = requested_keyframe_mode
        self.prestop_window = int(prestop_window)

        self.learned_keyframe_cache_path = learned_keyframe_cache
        self.learned_keyframe_cache: Dict[str, Dict[str, Any]] = {}

        if self.keyframe_mode == "learned_cache":
            if not learned_keyframe_cache:
                raise ValueError("keyframe_mode=learned_cache requires learned_keyframe_cache")

            p = Path(learned_keyframe_cache)
            if not p.exists():
                raise FileNotFoundError(f"Cannot find learned_keyframe_cache: {p}")

            with open(p, "r", encoding="utf-8") as f:
                self.learned_keyframe_cache = json.load(f)

            print(
                f"[OpenFlyFeatureDualDataset] loaded learned_keyframe_cache={p}, "
                f"records={len(self.learned_keyframe_cache)}",
                flush=True,
            )

        if self.keyframe_mode not in ["residual", "label", "learned_cache"]:
            raise ValueError(f"Unsupported keyframe_mode={self.keyframe_mode}")

    @staticmethod
    def _cache_key(image_path: str, cur_idx: int) -> str:
        return f"{image_path}::{int(cur_idx)}"

    def _label_fallback_keyframe_idx(self, ep: Dict[str, Any], cur_idx: int) -> int:
        if self.exclude_previous:
            end_exclusive = cur_idx - 1
        else:
            end_exclusive = cur_idx

        start = max(0, end_exclusive - self.max_history)
        candidate_indices = list(range(start, end_exclusive))

        if len(candidate_indices) == 0:
            return max(0, cur_idx - 2)

        scored = [(j, self._candidate_score(ep, j)) for j in candidate_indices]
        best_idx, _ = max(scored, key=lambda x: x[1])
        return int(best_idx)

    def _select_learned_cache_idx(
        self,
        ep: Dict[str, Any],
        ep_meta: Dict[str, Any],
        cur_idx: int,
    ) -> int:
        ann = ep_meta["annotation"]
        image_path = str(ann.get("image_path", ""))
        parquet_path = str(ep_meta.get("parquet_path", ""))

        possible_keys = [
            self._cache_key(image_path, cur_idx),
            self._cache_key(parquet_path, cur_idx),
        ]

        rec = None
        for k in possible_keys:
            if k in self.learned_keyframe_cache:
                rec = self.learned_keyframe_cache[k]
                break

        if rec is None:
            return self._label_fallback_keyframe_idx(ep, cur_idx)

        key_idx = int(rec.get("keyframe_idx", max(0, cur_idx - 2)))
        prev_idx = max(0, cur_idx - 1)

        if self.exclude_previous and key_idx == prev_idx and cur_idx >= 2:
            key_idx = max(0, cur_idx - 2)

        if key_idx >= cur_idx:
            key_idx = max(0, cur_idx - 2)

        if key_idx < 0:
            key_idx = 0

        return int(key_idx)

    def _build_trend_text(
        self,
        *,
        action_label: int,
        waypoint_raw: torch.Tensor,
        stop_label: float,
        prestop_label: float,
        steps_to_end: int,
    ) -> str:
        d = float(waypoint_raw[0].item())
        yaw = float(waypoint_raw[1].item())
        z = float(waypoint_raw[2].item())

        if stop_label > 0.5:
            temporal_stop = "stable_stop_candidate"
        elif prestop_label > 0.5:
            temporal_stop = "near_target_but_continue_checking"
        else:
            temporal_stop = "continue_moving"

        return (
            " Auxiliary navigation trend state: "
            f"distance={_bucket_distance(d)}; "
            f"yaw={_bucket_yaw(yaw)}; "
            f"vertical={_bucket_z(z)}; "
            f"motion_hint={ACTION_NAMES.get(int(action_label), 'unknown')}; "
            f"prestop={int(prestop_label > 0.5)}; "
            f"temporal_stop_trend={temporal_stop}; "
            f"steps_to_end_bucket={'terminal' if steps_to_end <= 1 else ('near' if steps_to_end <= self.horizon else 'far')}. "
            "Use these trend states only as auxiliary context. "
            "Preserve the full movement action space and decide the final action yourself."
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_id, cur_idx = self.samples[idx]
        ep_meta = self.episodes[ep_id]
        ep = self._load_episode(ep_meta["parquet_path"])

        rows = ep["rows"]
        ann = ep_meta["annotation"]

        if self.keyframe_mode == "learned_cache":
            key_idx = self._select_learned_cache_idx(ep, ep_meta, cur_idx)
        else:
            key_idx = self._select_keyframe_idx(ep, cur_idx)

        prev_idx = max(0, cur_idx - 1)

        key_img = normalize_image_obj(rows[key_idx]["image"])
        prev_img = normalize_image_obj(rows[prev_idx]["image"])
        cur_img = normalize_image_obj(rows[cur_idx]["image"])

        waypoint_label, waypoint_raw = self._build_waypoint_label(rows, cur_idx)
        action_label = int(ep["action_ids"][cur_idx])

        steps_to_end = max(0, (len(rows) - 1) - cur_idx)

        # stop_label is strict stop commit; prestop_label is broad near-terminal trend.
        stop_label = 1.0 if (action_label == 0 or steps_to_end <= 1) else 0.0
        prestop_label = 1.0 if steps_to_end <= (self.horizon + self.prestop_window) else 0.0
        prestop_label = max(prestop_label, stop_label)

        original_instruction = str(ann.get("gpt_instruction", "")).lower()
        trend_text = self._build_trend_text(
            action_label=action_label,
            waypoint_raw=waypoint_raw,
            stop_label=stop_label,
            prestop_label=prestop_label,
            steps_to_end=steps_to_end,
        )
        instruction_with_trend = original_instruction + trend_text

        return {
            "images": [key_img, prev_img, cur_img],
            "instruction": instruction_with_trend,
            "original_instruction": original_instruction,
            "trend_text": trend_text,
            "action_label": torch.tensor(action_label, dtype=torch.long),
            "stop_label": torch.tensor(stop_label, dtype=torch.float32),
            "prestop_label": torch.tensor(prestop_label, dtype=torch.float32),
            "temporal_stop_label": torch.tensor(stop_label, dtype=torch.float32),
            "waypoint_label": waypoint_label,
            "waypoint_raw": waypoint_raw,
            "waypoint_mask": torch.tensor(1.0, dtype=torch.float32),
            "meta": {
                "parquet_path": ep_meta["parquet_path"],
                "image_path": ann.get("image_path", ""),
                "current_idx": cur_idx,
                "future_idx": cur_idx + self.horizon,
                "keyframe_idx": key_idx,
                "previous_idx": prev_idx,
                "steps_to_end": steps_to_end,
                "keyframe_mode": self.keyframe_mode,
            },
        }


def collate_openfly_feature_dual_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": [x["images"] for x in batch],
        "instruction": [x["instruction"] for x in batch],
        "original_instruction": [x.get("original_instruction", "") for x in batch],
        "trend_text": [x.get("trend_text", "") for x in batch],
        "action_label": torch.stack([x["action_label"] for x in batch], dim=0),
        "stop_label": torch.stack([x["stop_label"] for x in batch], dim=0),
        "prestop_label": torch.stack([x["prestop_label"] for x in batch], dim=0),
        "temporal_stop_label": torch.stack([x["temporal_stop_label"] for x in batch], dim=0),
        "waypoint_label": torch.stack([x["waypoint_label"] for x in batch], dim=0),
        "waypoint_raw": torch.stack([x["waypoint_raw"] for x in batch], dim=0),
        "waypoint_mask": torch.stack([x["waypoint_mask"] for x in batch], dim=0),
        "meta": [x["meta"] for x in batch],
    }
