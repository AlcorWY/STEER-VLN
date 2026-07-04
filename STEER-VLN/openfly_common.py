import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader


def get_eval_image_dir(root="images", baseline_method="openfly", env_name="unknown", sample_idx: int = 0):
    out_dir = Path(root) / "eval_baseline" / baseline_method / env_name / f"sample_{sample_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_eval_task_summary(
    image_dir,
    instruction,
    env_name,
    sample_idx,
    baseline_method,
    start_position,
    goal_position,
    start_yaw,
    final_position,
    success,
    osr,
    num_steps,
    action_sequence,
    image_files,
    stop_pred,
    image_error,
    additional_info=None,
):
    if isinstance(image_dir, (str, Path)):
        image_dir = Path(image_dir)

    summary = {
        "baseline_method": baseline_method,
        "env_name": env_name,
        "sample_idx": sample_idx,
        "instruction": instruction,
        "start_position": start_position,
        "goal_position": goal_position,
        "start_yaw": start_yaw,
        "final_position": final_position,
        "success": int(success),
        "osr": int(osr),
        "num_steps": num_steps,
        "action_sequence": action_sequence,
        "image_files": image_files,
        "predicted_stop": bool(stop_pred),
        "image_error": bool(image_error),
        "additional_info": additional_info or {},
    }

    summary_path = image_dir / "task_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    root_summary = image_dir.parents[2] / "summary.jsonl"
    root_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(root_summary, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def _find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur.parent] + list(cur.parents):
        if (parent / "configs").exists() and (parent / "code").exists() and (parent / "requirements.txt").exists():
            return parent
    raise RuntimeError("Cannot locate OpenFly-Platform root")


ROOT = _find_project_root()
STEER_VLN_DIR = ROOT / "STEER-VLN"
CODE_DIR = ROOT / "code"
TRAIN_DIR = ROOT / "train"

# STEER_VLN priority:
#   STEER-VLN -> code -> project root -> train
# train/ is appended last to avoid shadowing HuggingFace datasets by train/datasets.
for pp in [str(STEER_VLN_DIR), str(CODE_DIR), str(ROOT), str(TRAIN_DIR)]:
    if pp in sys.path:
        sys.path.remove(pp)

sys.path.insert(0, str(STEER_VLN_DIR))
sys.path.insert(1, str(CODE_DIR))
sys.path.insert(2, str(ROOT))
sys.path.append(str(TRAIN_DIR))

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from extern.hf.configuration_prismatic import OpenFlyConfig
from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def set_seed(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def register_openfly_hf_classes():
    AutoConfig.register("openvla", OpenFlyConfig)
    AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)


def _resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((ROOT / p).resolve())


def load_openfly_local(
    model_path: str = "models/openfly-agent-7b",
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
):
    """
    STEER_VLN local-only OpenFly loader.
    """
    register_openfly_hf_classes()
    model_path = _resolve_path(model_path)

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    policy = AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)

    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    print(f"[Load OpenFly] local model: {model_path}", flush=True)
    print("[Load OpenFly] frozen backbone", flush=True)

    return processor, policy


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    collate_fn,
    prefetch_factor: int = 2,
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
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True

    return DataLoader(**kwargs)


def move_labels_to_device(batch: Dict, device: torch.device):
    out = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v

    return out


def _move_processor_inputs(inputs, device: torch.device, dtype: torch.dtype):
    moved = {}

    for k, v in inputs.items():
        if torch.is_tensor(v):
            if v.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                moved[k] = v.to(device=device, dtype=dtype, non_blocking=True)
            else:
                moved[k] = v.to(device=device, non_blocking=True)
        else:
            moved[k] = v

    return moved


def _last_token_feature_from_outputs(outputs, inputs, device: torch.device):
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        raise RuntimeError("OpenFly forward did not return hidden_states.")

    last_hidden = outputs.hidden_states[-1]

    if "attention_mask" in inputs:
        attn = inputs["attention_mask"]
        idx = attn.sum(dim=1).long() - 1
        feat = last_hidden[torch.arange(last_hidden.shape[0], device=device), idx]
    else:
        feat = last_hidden[:, -1]

    return feat[0].float()


@torch.no_grad()
def extract_openfly_features(
    policy,
    pixel_values=None,
    processor=None,
    instructions=None,
    images_batch=None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    STEER_VLN batch-safe OpenFly feature extraction.

    Supports:
      1. old style:
         extract_openfly_features(policy, pixel_values)

      2. STEER_VLN keyword style:
         extract_openfly_features(
             policy=policy,
             processor=processor,
             instructions=batch["instruction"],
             images_batch=batch["images"],
             device=device,
             dtype=dtype,
         )

    Handles:
      - images_batch = list[list[PIL.Image]]
      - images_batch = list[PIL.Image] for one sample
      - pixel_values tensor
    """
    if device is None:
        try:
            device = next(policy.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Case 1: direct tensor pixel_values.
    # ------------------------------------------------------------------
    if pixel_values is not None and torch.is_tensor(pixel_values):
        x = pixel_values.to(device=device, non_blocking=True)
        if x.is_floating_point():
            x = x.to(dtype=dtype)

        if hasattr(policy, "vision_backbone"):
            try:
                feat = policy.vision_backbone(x)
            except TypeError:
                feat = policy.vision_backbone(pixel_values=x)
        elif hasattr(policy, "get_image_features"):
            feat = policy.get_image_features(x)
        else:
            raise AttributeError("Cannot extract direct pixel_values features from policy.")

        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if isinstance(feat, dict):
            for key in ["pooler_output", "last_hidden_state", "image_features"]:
                if key in feat:
                    feat = feat[key]
                    break

        if hasattr(feat, "last_hidden_state"):
            feat = feat.last_hidden_state

        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output

        if feat.dim() == 3:
            feat = feat.mean(dim=1)

        if feat.dim() > 2:
            feat = feat.flatten(start_dim=1)

        return feat.float()

    # ------------------------------------------------------------------
    # Case 2: STEER_VLN normal path, list/list image batch.
    # ------------------------------------------------------------------
    if images_batch is None:
        raise ValueError("images_batch is required when pixel_values is None.")

    if processor is None:
        raise ValueError("processor is required for images_batch feature extraction.")

    if instructions is None:
        if isinstance(images_batch, (list, tuple)):
            instructions = [""] * len(images_batch)
        elif torch.is_tensor(images_batch):
            instructions = [""] * int(images_batch.shape[0])
        else:
            instructions = [""]

    # Tensor batch fallback.
    if torch.is_tensor(images_batch):
        x = images_batch

        if x.dim() == 5:
            b, t = x.shape[:2]
            x = x.reshape(b * t, *x.shape[2:])
            feat = extract_openfly_features(
                policy=policy,
                pixel_values=x,
                device=device,
                dtype=dtype,
            )
            return feat.reshape(b, t, -1).mean(dim=1).float()

        if x.dim() == 4:
            return extract_openfly_features(
                policy=policy,
                pixel_values=x,
                device=device,
                dtype=dtype,
            )

        raise ValueError(f"Unsupported tensor images_batch shape: {tuple(x.shape)}")

    def _encode_one(instr, imgs):
        # imgs should be one sample: [keyframe, previous, current].
        text = str(instr).lower()

        try:
            encoded = processor(text, imgs, return_tensors="pt")
        except TypeError:
            try:
                encoded = processor(
                    text=text,
                    images=imgs,
                    return_tensors="pt",
                    padding=True,
                )
            except TypeError:
                encoded = processor(
                    images=imgs,
                    return_tensors="pt",
                )

        if not isinstance(encoded, dict):
            encoded = dict(encoded)

        inputs = _move_processor_inputs(encoded, device=device, dtype=dtype)

        outputs = policy(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        return _last_token_feature_from_outputs(outputs, inputs, device)

    feats = []

    if isinstance(images_batch, (list, tuple)):
        if len(images_batch) == 0:
            raise ValueError("images_batch is empty.")

        # batch: [[keyframe, previous, current], ...]
        if isinstance(images_batch[0], (list, tuple)):
            for instr, imgs in zip(instructions, images_batch):
                feats.append(_encode_one(instr, imgs))
        else:
            # one sample: [keyframe, previous, current]
            instr = instructions[0] if isinstance(instructions, (list, tuple)) else instructions
            feats.append(_encode_one(instr, images_batch))

        return torch.stack(feats, dim=0)

    raise TypeError(f"Unsupported images_batch type: {type(images_batch)}")


def aggregate_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}

    keys = items[0].keys()
    out = {}

    for k in keys:
        vals = [float(x[k]) for x in items if k in x]
        if vals:
            out[k] = float(sum(vals) / len(vals))

    return out
