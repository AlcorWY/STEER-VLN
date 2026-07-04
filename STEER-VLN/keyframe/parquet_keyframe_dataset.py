import json
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
import pyarrow.parquet as pq


ACTION_ID_TO_VEC = {
    0: [1, 0, 0, 0, 0, 0, 0, 0],
    1: [0, 3, 0, 0, 0, 0, 0, 0],
    2: [0, 0, 15, 0, 0, 0, 0, 0],
    3: [0, 0, 0, 15, 0, 0, 0, 0],
    4: [0, 0, 0, 0, 2, 0, 0, 0],
    5: [0, 0, 0, 0, 0, 2, 0, 0],
    6: [0, 0, 0, 0, 0, 0, 5, 0],
    7: [0, 0, 0, 0, 0, 0, 0, 5],
    8: [0, 6, 0, 0, 0, 0, 0, 0],
    9: [0, 9, 0, 0, 0, 0, 0, 0],
}

TURN_ACTION_IDS = {2, 3, 4, 5, 6, 7}


def action_type_value_to_id(action_type: str, action_value: int) -> int:
    """
    Robust mapping from OpenFly parquet action_type/action_value to discrete action id.

    Handles known typos and value inconsistencies:
      - "donw" -> down
      - "go down" with value 0 -> action 5
      - "go up" with value 0 -> action 4
    """
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


def wrap_angle(x: float) -> float:
    while x > math.pi:
        x -= 2 * math.pi
    while x < -math.pi:
        x += 2 * math.pi
    return x


def normalize_image_obj(img) -> Image.Image:
    """
    HuggingFace datasets.Image() may return:
      1. PIL.Image.Image
      2. {"bytes": ..., "path": ...}
      3. {"path": ...}
    Convert all supported forms to PIL RGB image.
    """
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


def image_difference_score(a, b) -> float:
    """
    Lightweight visual-landmark proxy.

    This is intentionally optional because it is CPU-heavy:
      - PIL decode / convert
      - resize
      - numpy difference
    """
    a = normalize_image_obj(a).resize((64, 64))
    b = normalize_image_obj(b).resize((64, 64))

    aa = np.asarray(a).astype(np.float32) / 255.0
    bb = np.asarray(b).astype(np.float32) / 255.0

    global_diff = np.mean(np.abs(aa - bb))

    c0, c1 = 16, 48
    center_diff = np.mean(np.abs(aa[c0:c1, c0:c1] - bb[c0:c1, c0:c1]))

    return float(0.5 * global_diff + 0.5 * center_diff)


class ParquetKeyframeDataset(Dataset):
    """
    Map-style dataset for learned keyframe scorer.

    One sample:
      current step t
      previous K candidate history frames
      soft label from action-change / turn / stop / motion residual
      optional image_difference_score as weak landmark supervision
    """

    def __init__(
        self,
        annotation_path: str,
        parquet_root: str,
        tokenizer,
        max_history: int = 8,
        image_size: int = 224,
        max_episodes: Optional[int] = None,
        stride: int = 1,
        label_temperature: float = 0.7,
        min_timestep: int = 1,
        cache_size: int = 64,
        use_landmark_score: bool = False,
    ):
        super().__init__()

        self.annotation_path = Path(annotation_path)
        self.parquet_root = Path(parquet_root)
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.image_size = image_size
        self.label_temperature = label_temperature
        self.min_timestep = min_timestep
        self.cache_size = max(1, int(cache_size))
        self.use_landmark_score = bool(use_landmark_score)

        data = json.load(open(self.annotation_path, "r", encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError(f"annotation_path must be list json: {self.annotation_path}")

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

            if n <= self.min_timestep:
                continue

            ep_id = len(self.episodes)
            self.episodes.append(
                {
                    "annotation": item,
                    "parquet_path": str(parquet_path),
                    "num_rows": n,
                }
            )

            for t in range(self.min_timestep, n, stride):
                self.samples.append((ep_id, t))

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_order: List[str] = []

        print(
            f"[ParquetKeyframeDataset] episodes={len(self.episodes)}, "
            f"samples={len(self.samples)}, max_history={self.max_history}, "
            f"cache_size={self.cache_size}, use_landmark_score={self.use_landmark_score}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _read_rows_no_cache(self, parquet_path: str) -> List[Dict[str, Any]]:
        """
        Fast parquet reader for keyframe scorer/cache generation.
        Avoids repeated HuggingFace datasets "Generating train split".
        """
        table = pq.read_table(parquet_path)
        rows = table.to_pylist()
        rows = sorted(rows, key=lambda r: int(r["frame_index"]))
        return rows

    def _read_num_rows_fast(self, parquet_path: str) -> int:
        """
        Fast metadata-only row count.
        Avoids load_dataset() during Dataset initialization.
        """
        try:
            return int(pq.ParquetFile(parquet_path).metadata.num_rows)
        except Exception:
            # Fallback only when metadata reading fails.
            rows = self._read_rows_no_cache(parquet_path)
            return len(rows)

    def _get_action_ids(self, rows: List[Dict[str, Any]]) -> List[int]:
        ids = []
        for r in rows:
            ids.append(action_type_value_to_id(r["action_type"], r["action_value"]))
        return ids

    def _reference_points(self, action_ids: List[int]) -> Dict[str, List[int]]:
        change_points = []
        turn_points = []
        stop_points = []

        for i in range(1, len(action_ids)):
            if action_ids[i] != action_ids[i - 1]:
                change_points.append(i)

        for i, a in enumerate(action_ids):
            if a in TURN_ACTION_IDS:
                turn_points.append(i)
            if a == 0:
                stop_points.append(i)

        return {
            "change": sorted(set(change_points)),
            "turn": sorted(set(turn_points)),
            "stop": sorted(set(stop_points)),
        }

    def _load_episode(self, parquet_path: str) -> Dict[str, Any]:
        if parquet_path in self._cache:
            return self._cache[parquet_path]

        rows = self._read_rows_no_cache(parquet_path)
        action_ids = self._get_action_ids(rows)
        refs = self._reference_points(action_ids)

        episode = {
            "rows": rows,
            "action_ids": action_ids,
            "refs": refs,
        }

        self._cache[parquet_path] = episode
        self._cache_order.append(parquet_path)

        while len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

        return episode

    def _motion_magnitude(self, rows: List[Dict[str, Any]], idx: int) -> float:
        if idx <= 0:
            return 0.0

        p0 = rows[idx - 1]["pos"]
        p1 = rows[idx]["pos"]
        y0 = float(rows[idx - 1]["yaw"])
        y1 = float(rows[idx]["yaw"])

        dp = np.linalg.norm(np.asarray(p1, dtype=np.float32) - np.asarray(p0, dtype=np.float32))
        dyaw = abs(wrap_angle(y1 - y0))

        return float(dp / 9.0 + dyaw / math.pi)

    def _motion_feat(self, rows: List[Dict[str, Any]], cand_idx: int, cur_idx: int) -> List[float]:
        pc = np.asarray(rows[cur_idx]["pos"], dtype=np.float32)
        ph = np.asarray(rows[cand_idx]["pos"], dtype=np.float32)

        dx, dy, dz = (pc - ph).tolist()
        dyaw = wrap_angle(float(rows[cur_idx]["yaw"]) - float(rows[cand_idx]["yaw"]))

        return [
            float(dx / 50.0),
            float(dy / 50.0),
            float(dz / 50.0),
            float(dyaw / math.pi),
        ]

    def _build_soft_label(
        self,
        episode: Dict[str, Any],
        candidate_indices: List[int],
        current_idx: int,
    ) -> torch.Tensor:
        rows = episode["rows"]
        refs = episode["refs"]

        scores = []
        current_img = rows[current_idx]["image"] if self.use_landmark_score else None

        motion_values = [self._motion_magnitude(rows, j) for j in candidate_indices]
        max_motion = max(motion_values) if motion_values else 1.0
        max_motion = max(max_motion, 1e-6)

        change_points = refs["change"]
        turn_points = refs["turn"]
        stop_points = refs["stop"]

        for local_i, j in enumerate(candidate_indices):
            # 1. Near action-change / turn point.
            turn_score = 0.0
            for p in change_points + turn_points:
                d = abs(j - p)
                turn_score = max(turn_score, math.exp(-(d * d) / (2 * 2.0 * 2.0)))

            # 2. Near stop point.
            stop_score = 0.0
            for p in stop_points:
                d = abs(j - p)
                stop_score = max(stop_score, math.exp(-(d * d) / (2 * 3.0 * 3.0)))

            # 3. Motion residual score.
            motion_score = motion_values[local_i] / max_motion

            # 4. Optional visual-landmark proxy.
            if self.use_landmark_score:
                try:
                    landmark_score = image_difference_score(rows[j]["image"], current_img)
                except Exception:
                    landmark_score = 0.0

                score = (
                    0.45 * turn_score
                    + 0.20 * stop_score
                    + 0.20 * motion_score
                    + 0.15 * landmark_score
                )
            else:
                score = (
                    0.55 * turn_score
                    + 0.20 * stop_score
                    + 0.25 * motion_score
                )

            scores.append(score)

        s = torch.tensor(scores, dtype=torch.float32)

        if s.numel() == 0:
            return s

        if torch.all(s <= 1e-8):
            s = torch.ones_like(s)

        s = torch.softmax(s / max(self.label_temperature, 1e-6), dim=0)
        return s

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_id, cur_idx = self.samples[idx]
        ep = self.episodes[ep_id]
        episode = self._load_episode(ep["parquet_path"])

        rows = episode["rows"]
        ann = ep["annotation"]
        lang = str(ann.get("gpt_instruction", "")).lower()

        start = max(0, cur_idx - self.max_history)
        candidate_indices = list(range(start, cur_idx))

        if len(candidate_indices) == 0:
            candidate_indices = [0]

        history_tensors = []
        motion_feats = []
        history_mask = []

        for j in candidate_indices:
            history_tensors.append(pil_to_tensor(rows[j]["image"], self.image_size))
            motion_feats.append(self._motion_feat(rows, j, cur_idx))
            history_mask.append(True)

        while len(history_tensors) < self.max_history:
            history_tensors.insert(0, torch.zeros(3, self.image_size, self.image_size))
            motion_feats.insert(0, [0.0, 0.0, 0.0, 0.0])
            history_mask.insert(0, False)

        if len(history_tensors) > self.max_history:
            history_tensors = history_tensors[-self.max_history:]
            motion_feats = motion_feats[-self.max_history:]
            history_mask = history_mask[-self.max_history:]

        current_image = pil_to_tensor(rows[cur_idx]["image"], self.image_size)

        soft = self._build_soft_label(episode, candidate_indices, cur_idx)

        padded_soft = torch.zeros(self.max_history, dtype=torch.float32)
        if soft.numel() > 0:
            padded_soft[-soft.numel():] = soft

        encoded = self.tokenizer(
            lang,
            add_special_tokens=True,
            truncation=True,
            max_length=128,
            padding=False,
            return_attention_mask=True,
        )

        return {
            "history_images": torch.stack(history_tensors, dim=0),
            "current_image": current_image,
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "motion_feats": torch.tensor(motion_feats, dtype=torch.float32),
            "history_mask": torch.tensor(history_mask, dtype=torch.bool),
            "soft_labels": padded_soft,
            "meta": {
                "parquet_path": ep["parquet_path"],
                "current_idx": cur_idx,
                "candidate_indices": candidate_indices,
                "image_path": ann.get("image_path", ""),
            },
        }


def collate_keyframe_batch(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    max_len = max(x["input_ids"].numel() for x in batch)

    input_ids = []
    attention_mask = []

    for x in batch:
        ids = x["input_ids"]
        mask = x["attention_mask"]
        pad_len = max_len - ids.numel()

        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)], dim=0)

        input_ids.append(ids)
        attention_mask.append(mask)

    return {
        "history_images": torch.stack([x["history_images"] for x in batch], dim=0),
        "current_image": torch.stack([x["current_image"] for x in batch], dim=0),
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "motion_feats": torch.stack([x["motion_feats"] for x in batch], dim=0),
        "history_mask": torch.stack([x["history_mask"] for x in batch], dim=0),
        "soft_labels": torch.stack([x["soft_labels"] for x in batch], dim=0),
        "meta": [x["meta"] for x in batch],
    }