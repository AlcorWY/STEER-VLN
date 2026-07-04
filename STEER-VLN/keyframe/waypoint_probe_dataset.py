import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
# ===== STEER_VLN guard: avoid local train/datasets shadowing HuggingFace datasets =====
import sys as _steer_vln_sys
from pathlib import Path as _STEER_VLNPath

_STEER_VLN_CUR = _STEER_VLNPath(__file__).resolve()
_STEER_VLN_ROOT = None
for _parent in [_STEER_VLN_CUR.parent] + list(_STEER_VLN_CUR.parents):
    if (_parent / "configs").exists() and (_parent / "code").exists() and (_parent / "requirements.txt").exists():
        _STEER_VLN_ROOT = _parent
        break

if _STEER_VLN_ROOT is not None:
    _STEER_VLN_TRAIN_DIR = str((_STEER_VLN_ROOT / "train").resolve())

    def _steer_vln_norm_path(x):
        try:
            return str(_STEER_VLNPath(x).resolve())
        except Exception:
            return str(x)

    _steer_vln_sys.path[:] = [
        x for x in _steer_vln_sys.path
        if _steer_vln_norm_path(x) != _STEER_VLN_TRAIN_DIR
    ]

    _datasets_mod = _steer_vln_sys.modules.get("datasets")
    _datasets_file = str(getattr(_datasets_mod, "__file__", "")) if _datasets_mod is not None else ""
    if "/train/datasets" in _datasets_file:
        del _steer_vln_sys.modules["datasets"]

from datasets import load_dataset
# ===== end STEER_VLN guard =====


TURN_ACTION_IDS = {2, 3, 4, 5, 6, 7}


def wrap_angle(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def normalize_image_obj(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, dict):
        if img.get("bytes") is not None:
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if img.get("path") is not None:
            return Image.open(img["path"]).convert("RGB")

    raise TypeError(f"Unsupported image object type: {type(img)}, value={img}")


def pil_to_tensor(img, image_size: int = 224) -> torch.Tensor:
    img = normalize_image_obj(img).resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).float()


def action_type_value_to_id(action_type: str, action_value: int) -> int:
    t = str(action_type).strip().lower()
    v = int(action_value)

    aliases = {
        "donw": "down",
        "dowm": "down",
        "go donw": "go down",
        "go dowm": "go down",
        "descend": "down",
        "ascend": "up",
        "straight": "go straight",
        "forward": "go straight",
        "move forward": "go straight",
        "left turn": "turn left",
        "right turn": "turn right",
    }
    t = aliases.get(t, t)

    if t in ["stop", "arrive", "finish"]:
        return 0

    if t == "go straight":
        if v == 3:
            return 1
        if v == 6:
            return 8
        if v == 9:
            return 9
        if v <= 3:
            return 1
        if v <= 6:
            return 8
        return 9

    if t == "turn left":
        return 2
    if t == "turn right":
        return 3
    if t in ["go up", "up"]:
        return 4
    if t in ["go down", "down"]:
        return 5
    if t in ["move left", "left"]:
        return 6
    if t in ["move right", "right"]:
        return 7

    raise ValueError(f"Unknown action_type/action_value: {action_type} / {action_value}")


class WaypointProbeDataset(Dataset):
    """
    Lightweight dataset for Stage 3A waypoint probe.

    It does NOT load or train OpenFly-Agent 7B.

    Each sample:
        images: [3, 3, H, W]
            order = [keyframe, previous, current]

        input_ids / attention_mask:
            instruction tokens from SimpleTextTokenizer

        motion_feats: [8]
            keyframe-current relative pose feat + previous-current relative pose feat

        waypoint_label: [3]
            [Δd / d_scale, Δyaw / pi, Δz / z_scale]

        waypoint_raw: [3]
            [Δd, Δyaw, Δz]
    """

    def __init__(
        self,
        annotation_path: str,
        parquet_root: str,
        tokenizer,
        image_size: int = 224,
        max_history: int = 8,
        horizon: int = 2,
        stride: int = 1,
        max_episodes: Optional[int] = None,
        keyframe_mode: str = "label",
        exclude_previous: bool = True,
        d_scale: float = 9.0,
        z_scale: float = 5.0,
        cache_size: int = 64,
        min_timestep: int = 2,
    ):
        super().__init__()

        self.annotation_path = Path(annotation_path)
        self.parquet_root = Path(parquet_root)
        self.tokenizer = tokenizer
        self.image_size = int(image_size)
        self.max_history = int(max_history)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.keyframe_mode = keyframe_mode
        self.exclude_previous = bool(exclude_previous)
        self.d_scale = float(d_scale)
        self.z_scale = float(z_scale)
        self.cache_size = max(1, int(cache_size))
        self.min_timestep = int(min_timestep)

        if self.keyframe_mode not in ["residual", "label"]:
            raise ValueError(f"Unsupported keyframe_mode={self.keyframe_mode}")

        data = json.load(open(self.annotation_path, "r", encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError(f"annotation_path must be a list json: {self.annotation_path}")

        if max_episodes is not None and max_episodes > 0:
            data = data[:max_episodes]

        self.episodes: List[Dict[str, Any]] = []
        self.samples: List[Tuple[int, int]] = []

        for item in data:
            image_path = item.get("image_path", "")
            if not image_path:
                continue

            parquet_path = self.parquet_root / Path(image_path).with_suffix(".parquet")
            if not parquet_path.exists():
                continue

            try:
                n = self._read_num_rows_fast(str(parquet_path))
            except Exception as e:
                print(f"[WARN] skip bad parquet metadata: {parquet_path}, error={e}")
                continue

            # Need t >= min_timestep and t + horizon < n
            if n <= self.min_timestep + self.horizon:
                continue

            ep_id = len(self.episodes)
            self.episodes.append(
                {
                    "annotation": item,
                    "parquet_path": str(parquet_path),
                    "num_rows": n,
                }
            )

            max_t = n - self.horizon - 1
            for t in range(self.min_timestep, max_t + 1, self.stride):
                self.samples.append((ep_id, t))

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_order: List[str] = []

        print(
            f"[WaypointProbeDataset] episodes={len(self.episodes)}, "
            f"samples={len(self.samples)}, image_size={self.image_size}, "
            f"horizon={self.horizon}, max_history={self.max_history}, "
            f"keyframe_mode={self.keyframe_mode}, exclude_previous={self.exclude_previous}, "
            f"cache_size={self.cache_size}"
        )

    def __len__(self):
        return len(self.samples)

    def _read_num_rows_fast(self, parquet_path: str) -> int:
        return int(pq.ParquetFile(parquet_path).metadata.num_rows)

    def _read_rows_no_cache(self, parquet_path: str) -> List[Dict[str, Any]]:
        """
        Fast parquet reader.

        Avoid datasets.load_dataset("parquet") here because it repeatedly prints
        "Generating train split" and adds heavy dataset-building overhead on cache miss.
        """
        table = pq.read_table(parquet_path)
        rows = table.to_pylist()
        rows = sorted(rows, key=lambda r: int(r["frame_index"]))
        return rows

    def _load_episode(self, parquet_path: str) -> Dict[str, Any]:
        if parquet_path in self._cache:
            return self._cache[parquet_path]

        rows = self._read_rows_no_cache(parquet_path)

        action_ids = []
        for r in rows:
            action_ids.append(action_type_value_to_id(r["action_type"], r["action_value"]))

        refs = self._reference_points(action_ids)

        ep = {
            "rows": rows,
            "action_ids": action_ids,
            "refs": refs,
        }

        self._cache[parquet_path] = ep
        self._cache_order.append(parquet_path)

        while len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

        return ep

    def _reference_points(self, action_ids: List[int]) -> Dict[str, List[int]]:
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

    def _motion_magnitude(self, rows: List[Dict[str, Any]], idx: int) -> float:
        if idx <= 0:
            return 0.0

        p0 = np.asarray(rows[idx - 1]["pos"], dtype=np.float32)
        p1 = np.asarray(rows[idx]["pos"], dtype=np.float32)
        y0 = float(rows[idx - 1]["yaw"])
        y1 = float(rows[idx]["yaw"])

        dp = np.linalg.norm(p1 - p0)
        dyaw = abs(wrap_angle(y1 - y0))

        return float(dp / 9.0 + dyaw / math.pi)

    def _candidate_score(
        self,
        ep: Dict[str, Any],
        candidate_idx: int,
    ) -> float:
        rows = ep["rows"]
        refs = ep["refs"]

        turn_score = 0.0
        for p in refs["change"] + refs["turn"]:
            d = abs(candidate_idx - p)
            turn_score = max(turn_score, math.exp(-(d * d) / (2 * 2.0 * 2.0)))

        stop_score = 0.0
        for p in refs["prestop"]:
            d = abs(candidate_idx - p)
            stop_score = max(stop_score, math.exp(-(d * d) / (2 * 3.0 * 3.0)))

        motion_score = self._motion_magnitude(rows, candidate_idx)

        # Same spirit as current no_imgdiff scorer label.
        return 0.55 * turn_score + 0.20 * stop_score + 0.25 * motion_score

    def _select_keyframe_idx(self, ep: Dict[str, Any], cur_idx: int) -> int:
        if self.keyframe_mode == "residual":
            return max(0, cur_idx - 2)

        # label mode: choose highest pseudo-keyframe score from history candidates.
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

    def _relative_motion_feat(
        self,
        rows: List[Dict[str, Any]],
        source_idx: int,
        cur_idx: int,
    ) -> List[float]:
        pc = np.asarray(rows[cur_idx]["pos"], dtype=np.float32)
        ps = np.asarray(rows[source_idx]["pos"], dtype=np.float32)

        dx, dy, dz = (pc - ps).tolist()
        dyaw = wrap_angle(float(rows[cur_idx]["yaw"]) - float(rows[source_idx]["yaw"]))

        return [
            float(dx / 50.0),
            float(dy / 50.0),
            float(dz / 50.0),
            float(dyaw / math.pi),
        ]

    def _build_waypoint_label(
        self,
        rows: List[Dict[str, Any]],
        cur_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fut_idx = cur_idx + self.horizon

        p0 = np.asarray(rows[cur_idx]["pos"], dtype=np.float32)
        p1 = np.asarray(rows[fut_idx]["pos"], dtype=np.float32)
        yaw0 = float(rows[cur_idx]["yaw"])

        dx, dy, dz = (p1 - p0).tolist()

        d = math.sqrt(dx * dx + dy * dy)

        if d > 1e-6:
            target_yaw = math.atan2(dy, dx)
        else:
            target_yaw = float(rows[fut_idx]["yaw"])

        dyaw = wrap_angle(target_yaw - yaw0)

        raw = torch.tensor([d, dyaw, dz], dtype=torch.float32)

        norm = torch.tensor(
            [
                max(0.0, min(d / self.d_scale, 1.5)),
                dyaw / math.pi,
                max(-1.5, min(dz / self.z_scale, 1.5)),
            ],
            dtype=torch.float32,
        )

        return norm, raw

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_id, cur_idx = self.samples[idx]
        ep_meta = self.episodes[ep_id]
        ep = self._load_episode(ep_meta["parquet_path"])

        rows = ep["rows"]
        ann = ep_meta["annotation"]

        key_idx = self._select_keyframe_idx(ep, cur_idx)
        prev_idx = max(0, cur_idx - 1)

        key_img = pil_to_tensor(rows[key_idx]["image"], self.image_size)
        prev_img = pil_to_tensor(rows[prev_idx]["image"], self.image_size)
        cur_img = pil_to_tensor(rows[cur_idx]["image"], self.image_size)

        # Order matches learned eval bundle:
        # [keyframe, previous, current]
        images = torch.stack([key_img, prev_img, cur_img], dim=0)

        motion_key = self._relative_motion_feat(rows, key_idx, cur_idx)
        motion_prev = self._relative_motion_feat(rows, prev_idx, cur_idx)
        motion_feats = torch.tensor(motion_key + motion_prev, dtype=torch.float32)

        waypoint_label, waypoint_raw = self._build_waypoint_label(rows, cur_idx)

        instruction = str(ann.get("gpt_instruction", "")).lower()

        encoded = self.tokenizer(
            instruction,
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            padding=False,
            return_attention_mask=True,
        )

        return {
            "images": images,
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "motion_feats": motion_feats,
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
            },
        }


def collate_waypoint_probe_batch(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    max_len = max(x["input_ids"].numel() for x in batch)

    input_ids = []
    attention_mask = []

    for x in batch:
        ids = x["input_ids"]
        mask = x["attention_mask"]

        pad_len = max_len - ids.numel()

        if pad_len > 0:
            ids = torch.cat(
                [ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)],
                dim=0,
            )
            mask = torch.cat(
                [mask, torch.zeros(pad_len, dtype=torch.long)],
                dim=0,
            )

        input_ids.append(ids)
        attention_mask.append(mask)

    return {
        "images": torch.stack([x["images"] for x in batch], dim=0),
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "motion_feats": torch.stack([x["motion_feats"] for x in batch], dim=0),
        "waypoint_label": torch.stack([x["waypoint_label"] for x in batch], dim=0),
        "waypoint_raw": torch.stack([x["waypoint_raw"] for x in batch], dim=0),
        "waypoint_mask": torch.stack([x["waypoint_mask"] for x in batch], dim=0),
        "meta": [x["meta"] for x in batch],
    }
