#!/usr/bin/env python3
"""
Unified resource-constrained OpenFly baseline evaluator for RTX 3090.

Put this file into OpenFly-Platform/train/ and run from the repo root:
    BASELINE_METHOD=random python train/eval_unified_baseline_3090.py --eval-info configs/eval_test.json --env-filter env_airsim_16 --limit-per-env 5

Supported methods:
    random                  no model, online OpenFly metric loop
    openfly                 official OpenFly-Agent checkpoint, direct eval
    openvla_direct          OpenVLA/OpenVLA-like direct-transfer adapter, experimental
    spf_api                 See-Point-Fly-style training-free VLM prompt via OpenAI-compatible API
    navid_direct_api        NaVid-style video VLM direct-transfer prompt via OpenAI-compatible API
    navila_direct_api       NaVILA-style navigation VLA direct-transfer prompt via OpenAI-compatible API
    oracle_replay           debug upper-bound executor that replays GT actions; not a paper baseline
    seq2seq_lite            trained lightweight Seq2Seq-style offline baseline checkpoint
    cma_lite                trained lightweight CMA-style offline baseline checkpoint
    aerialvln_lite          trained lightweight AerialVLN-style offline baseline checkpoint

The script deliberately keeps OpenFly's online simulator/eval loop and only swaps the policy.
It writes per-episode JSONL and aggregate CSV/JSON summaries under runs/baselines_3090/<method>/.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as _dt
import gc
import io
import json
import math
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import requests
from PIL import Image

# Keep imports lazy where possible, but the OpenFly bridge/actions are reused from the official eval.py.
# When this script is executed as `python train/eval_unified_baseline_3090.py`, sys.path includes train/.
from eval import (  # type: ignore
    AirsimBridge,
    GSBridge,
    UEBridge,
    calculate_distance,
    convert_to_action_id,
    getPoseAfterMakeAction,
    kill_env_process,
)

ACTION_ID_TO_NAME = {
    0: "Stop",
    1: "Forward_3m",
    2: "Turn_Left_30deg",
    3: "Turn_Right_30deg",
    4: "Move_Up_3m",
    5: "Move_Down_3m",
    6: "Move_Left_3m",
    7: "Move_Right_3m",
    8: "Forward_6m",
    9: "Forward_9m",
}

ACTION_NAME_TO_ID = {
    "stop": 0,
    "arrive": 0,
    "arrived": 0,
    "forward": 1,
    "move forward": 1,
    "go forward": 1,
    "straight": 1,
    "forward_3m": 1,
    "turn left": 2,
    "left": 2,
    "turn_left": 2,
    "turn right": 3,
    "right": 3,
    "turn_right": 3,
    "up": 4,
    "move up": 4,
    "go up": 4,
    "ascend": 4,
    "down": 5,
    "move down": 5,
    "go down": 5,
    "descend": 5,
    "move left": 6,
    "strafe left": 6,
    "move right": 7,
    "strafe right": 7,
    "forward_6m": 8,
    "forward 6": 8,
    "forward_9m": 9,
    "forward 9": 9,
}


@dataclass
class EvalConfig:
    method: str
    eval_info: str
    output_dir: Path
    env_filter: Optional[List[str]]
    limit_total: Optional[int]
    limit_per_env: Optional[int]
    max_step: int
    seed: int
    device: str
    image_history: int
    pos_ratio_gs: float
    stop_threshold_m: float
    dry_run: bool
    save_images: bool
    force_six_actions: bool
    random_stop_prob: float
    random_actions: List[int]
    openfly_model_path: str
    openvla_model_path: str
    lite_ckpt: str
    unnorm_key: str
    fallback_unnorm_keys: List[str]
    api_base_url: str
    api_key: str
    api_model: str
    api_temperature: float
    api_timeout: int
    api_frames: int
    api_max_retries: int
    default_action_on_error: int


class BasePolicy:
    name = "base"
    uses_cuda = False

    def reset(self, instruction: str, episode: Dict[str, Any]) -> None:
        self.instruction = instruction
        self.episode = episode

    def act(self, image_history: List[np.ndarray], instruction: str, episode: Dict[str, Any], step: int, pose: List[float]) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass


class RandomPolicy(BasePolicy):
    name = "random"

    def __init__(self, actions: Sequence[int], stop_prob: float = 0.05, seed: int = 0):
        self.actions = [int(a) for a in actions]
        self.stop_prob = float(stop_prob)
        self.rng = random.Random(seed)

    def act(self, image_history: List[np.ndarray], instruction: str, episode: Dict[str, Any], step: int, pose: List[float]) -> int:
        # Paper Random: randomly chooses actions until stop. This variant makes Stop possible at every step.
        if 0 in self.actions and self.rng.random() < self.stop_prob:
            return 0
        non_stop = [a for a in self.actions if a != 0]
        if not non_stop:
            return 0
        return int(self.rng.choice(non_stop))


class OracleReplayPolicy(BasePolicy):
    name = "oracle_replay_debug"

    def act(self, image_history: List[np.ndarray], instruction: str, episode: Dict[str, Any], step: int, pose: List[float]) -> int:
        acts = episode.get("action", [])
        if step < len(acts):
            return int(acts[step])
        return 0


class OpenFlyPolicy(BasePolicy):
    name = "openfly"
    uses_cuda = True

    def __init__(self, model_path: str, device: str = "cuda:0", unnorm_key: str = "vln_norm", fallback_keys: Optional[List[str]] = None):
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from extern.hf.configuration_prismatic import OpenFlyConfig
        from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

        AutoConfig.register("openvla", OpenFlyConfig, exist_ok=True)
        AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor, exist_ok=True)
        AutoProcessor.register(OpenFlyConfig, PrismaticProcessor, exist_ok=True)
        AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction, exist_ok=True)

        self.torch = torch
        self.device = device
        self.unnorm_key = unnorm_key
        self.fallback_keys = fallback_keys or ["vln_norm", "vlnv1", "openfly", "bridge_orig"]
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.policy = AutoModelForVision2Seq.from_pretrained(
            model_path,
            attn_implementation=os.environ.get("OPENFLY_ATTN_IMPL", "flash_attention_2"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.policy.eval()

    def _prepare_images(self, image_history: List[np.ndarray]) -> List[Image.Image]:
        frames = image_history[-3:]
        if len(frames) == 0:
            raise RuntimeError("empty image_history")
        while len(frames) < 3:
            frames = [frames[0]] + frames
        images: List[Image.Image] = []
        for img in frames:
            if img.ndim == 3 and img.shape[-1] == 3:
                # OpenCV images are BGR in many bridges. The official eval passed raw arrays directly;
                # using RGB here is safer for HF processors.
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            images.append(Image.fromarray(img_rgb.astype(np.uint8)))
        return images

    def act(self, image_history: List[np.ndarray], instruction: str, episode: Dict[str, Any], step: int, pose: List[float]) -> int:
        images = self._prepare_images(image_history)
        inputs = self.processor(instruction, images).to(self.device, dtype=self.torch.bfloat16)
        errors = []
        keys = [self.unnorm_key] + [k for k in self.fallback_keys if k != self.unnorm_key]
        for key in keys:
            try:
                action = self.policy.predict_action(**inputs, unnorm_key=key, do_sample=False)
                action = np.asarray(action).round().astype(int)
                action_id = int(convert_to_action_id(action))
                return action_id
            except Exception as e:  # try next unnorm_key
                errors.append(f"{key}: {repr(e)}")
        raise RuntimeError("OpenFly predict_action failed for all unnorm keys: " + " | ".join(errors))

    def close(self) -> None:
        try:
            del self.policy
            del self.processor
            self.torch.cuda.empty_cache()
        except Exception:
            pass


class OpenVLADirectPolicy(BasePolicy):
    """Experimental direct-transfer adapter for OpenVLA-like checkpoints.

    OpenVLA was trained for robot manipulation actions, not OpenFly UAV actions. This policy is only
    for a clearly labeled direct-transfer baseline. It maps the continuous action vector to a UAV action
    by a simple heuristic.
    """

    name = "openvla_direct"
    uses_cuda = True

    def __init__(self, model_path: str, device: str = "cuda:0", unnorm_key: str = "bridge_orig"):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.device = device
        self.unnorm_key = unnorm_key
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.policy = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.policy.eval()

    def _map_continuous_to_uav(self, action: np.ndarray) -> int:
        a = np.asarray(action).astype(float).flatten()
        if a.size == 0:
            return 1
        # Heuristic: translation components often occupy the first 3 dims in manipulation VLAs.
        # Choose the largest absolute movement dimension, otherwise forward.
        if a.size >= 3:
            dx, dy, dz = a[0], a[1], a[2]
            vals = np.array([abs(dx), abs(dy), abs(dz)])
            if vals.max() < 1e-3:
                return 1
            k = int(vals.argmax())
            if k == 2:
                return 4 if dz > 0 else 5
            if k == 1:
                return 6 if dy > 0 else 7
            return 1
        return 1

    def act(self, image_history: List[np.ndarray], instruction: str, episode: Dict[str, Any], step: int, pose: List[float]) -> int:
        img = image_history[-1]
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img.astype(np.uint8))
        prompt = f"What action should the robot take to {instruction.lower()}?"
        inputs = self.processor(prompt, pil).to(self.device, dtype=self.torch.bfloat16)
        action = self.policy.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return int(self._map_continuous_to_uav(np.asarray(action)))

    def close(self) -> None:
        try:
            del self.policy
            del self.processor
            self.torch.cuda.empty_cache()
        except Exception:
            pass


class APIVLMPolicy(BasePolicy):
    """OpenAI-compatible VLM direct-eval policy.

    Use this for training-free baselines when full model training is not feasible:
      - spf_api: See-Point-Fly-style current-frame target/heading decision.
      - navid_direct_api: video-history VLM next-action decision.
      - navila_direct_api: high-level navigation command mapped to UAV action.

    Required env variables:
      OPENAI_API_KEY or API_KEY
      OPENAI_BASE_URL, e.g. https://api.openai.com/v1 or DashScope compatible endpoint
      VLM_MODEL, e.g. gpt-4.1, qwen-vl-plus, qwen3-vl-flash, etc.
    """

    def __init__(
        self,
        variant: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout: int = 120,
        frames: int = 3,
        retries: int = 2,
        default_action: int = 1,
    ):
        self.variant = variant
        self.name = variant
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        self.frames = int(frames)
        self.retries = int(retries)
        self.default_action = int(default_action)
        if not self.api_key:
            raise RuntimeError("Missing API key. Set OPENAI_API_KEY or API_KEY.")
        if not self.base_url:
            raise RuntimeError("Missing API base URL. Set OPENAI_BASE_URL.")
        if not self.model:
            raise RuntimeError("Missing VLM model name. Set VLM_MODEL.")

    def _encode_frame(self, frame: np.ndarray) -> str:
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame.astype(np.uint8))
        im.thumbnail((768, 768))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _system_prompt(self) -> str:
        base = (
            "You are an aerial vision-language navigation policy for a UAV. "
            "Choose exactly one next action from the OpenFly action space. "
            "Return JSON only, no markdown. Valid action_id values are: "
            "0 Stop, 1 Forward_3m, 2 Turn_Left_30deg, 3 Turn_Right_30deg, "
            "4 Move_Up_3m, 5 Move_Down_3m, 6 Move_Left_3m, 7 Move_Right_3m, "
            "8 Forward_6m, 9 Forward_9m. "
            "Use Stop only if the target described by the instruction appears reached."
        )
        if self.variant == "spf_api":
            return base + " Follow a See-Point-Fly style policy: identify the next visual target point in the current image, then choose the action that best moves toward it."
        if self.variant == "navid_direct_api":
            return base + " Follow a NaVid-style policy: use the sequence of recent video frames and the instruction to predict the next navigation action."
        if self.variant == "navila_direct_api":
            return base + " Follow a NaVILA-style policy: infer a high-level navigation command and map it to the UAV action space."
        return base

    def _user_text(self, instruction: str, step: int, pose: List[float]) -> str:
        return (
            f"Instruction: {instruction}\n"
            f"Step: {step}\n"
            f"Current pose estimate [x,y,z,yaw_rad]: {pose}\n"
            "Choose the next action. Output JSON in this exact schema: "
            '{"action_id": 1, "action_name": "Forward_3m", "reason": "short reason"}'
        )

    def _parse_action(self, text: str) -> int:
        text = text.strip()
        # Extract JSON object if model adds extra text.
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if "action_id" in obj:
                    aid = int(obj["action_id"])
                    if 0 <= aid <= 9:
                        return aid
                for key in ("action", "action_name", "next_action"):
                    if key in obj:
                        s = str(obj[key]).strip().lower().replace("-", "_")
                        s2 = s.replace("_", " ")
                        if s in ACTION_NAME_TO_ID:
                            return ACTION_NAME_TO_ID[s]
                        if s2 in ACTION_NAME_TO_ID:
                            return ACTION_NAME_TO_ID[s2]
            except Exception:
                pass
        low = text.lower()
        # Prefer stop only if explicit.
        for key, aid in ACTION_NAME_TO_ID.items():
            if key in low:
                return int(aid)
        return self.default_action

    def act(self, image_history: List[np.ndarray], instruction: str, episode: Dict[str, Any], step: int, pose: List[float]) -> int:
        frames = image_history[-max(1, self.frames):]
        content: List[Dict[str, Any]] = [{"type": "text", "text": self._user_text(instruction, step, pose)}]
        for i, frame in enumerate(frames):
            b64 = self._encode_frame(frame)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": 128,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = self.base_url + "/chat/completions"
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                action_id = self._parse_action(text)
                print(f"[API:{self.variant}] raw={text[:200]!r} -> action={action_id} {ACTION_ID_TO_NAME.get(action_id)}")
                return int(action_id)
            except Exception as e:
                last_err = e
                time.sleep(1 + attempt)
        print(f"[WARN] API VLM failed after retries: {last_err}; default action={self.default_action}")
        return self.default_action


class LiteCheckpointPolicy(BasePolicy):
    """Online policy for resource-constrained trained baselines.

    The checkpoint is produced by baseline_eval_3090/lite/train_lite_baseline.py.
    It is intentionally small enough for RTX 3090 and is used as Seq2Seq-lite,
    CMA-lite or AerialVLN-lite in the unified OpenFly online evaluator.
    """
    uses_cuda = True

    def __init__(self, ckpt_path: str, device: str = "cuda:0"):
        if not ckpt_path:
            raise RuntimeError("Missing --lite-ckpt / LITE_CKPT for lite baseline evaluation")
        lite_dir = Path(__file__).resolve().parents[1] / "baseline_eval_3090" / "lite"
        if not lite_dir.exists():
            lite_dir = Path.cwd() / "baseline_eval_3090" / "lite"
        sys.path.insert(0, str(lite_dir))
        from policy_lite_runtime import LiteOnlinePolicy  # type: ignore
        self.inner = LiteOnlinePolicy(ckpt_path, device=device)
        self.name = self.inner.model_kind

    def reset(self, instruction: str, episode: Dict[str, Any]) -> None:
        self.inner.reset(instruction, episode)

    def act(self, image_history: List[np.ndarray], instruction: str, episode: Dict[str, Any], step: int, pose: List[float]) -> int:
        return int(self.inner.act(image_history, instruction, episode, step, pose))

    def close(self) -> None:
        self.inner.close()


def parse_env_filter(s: Optional[str]) -> Optional[List[str]]:
    if s is None or s.strip() == "" or s.strip().lower() in {"all", "none"}:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def action_list_from_env(force_six: bool, value: Optional[str]) -> List[int]:
    if value:
        return [int(x) for x in value.split(",") if x.strip() != ""]
    return [0, 1, 2, 3, 4, 5] if force_six else list(range(10))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified OpenFly baseline evaluator for RTX 3090")
    p.add_argument("--method", default=os.environ.get("BASELINE_METHOD", "random"),
                   choices=["random", "oracle_replay", "openfly", "openvla_direct", "spf_api", "navid_direct_api", "navila_direct_api", "seq2seq_lite", "cma_lite", "aerialvln_lite"])
    p.add_argument("--eval-info", default=os.environ.get("EVAL_INFO", "configs/eval_test.json"))
    p.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", "runs/baselines_3090"))
    p.add_argument("--env-filter", default=os.environ.get("ENV_FILTER", "env_airsim_16,env_airsim_26"),
                   help="comma-separated env names/substrings; default only light AirSim scenes")
    p.add_argument("--limit-total", type=int, default=int(os.environ.get("LIMIT_TOTAL", "0")) or None)
    p.add_argument("--limit-per-env", type=int, default=int(os.environ.get("LIMIT_PER_ENV", "5")) or None)
    p.add_argument("--max-step", type=int, default=int(os.environ.get("MAX_STEP", "100")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    p.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    p.add_argument("--image-history", type=int, default=int(os.environ.get("IMAGE_HISTORY", "3")))
    p.add_argument("--pos-ratio-gs", type=float, default=float(os.environ.get("POS_RATIO_GS", "5.15")))
    p.add_argument("--stop-threshold-m", type=float, default=float(os.environ.get("STOP_THRESHOLD_M", "20.0")))
    p.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "0") == "1")
    p.add_argument("--save-images", action="store_true", default=os.environ.get("SAVE_IMAGES", "0") == "1")
    p.add_argument("--force-six-actions", action="store_true", default=os.environ.get("FORCE_SIX_ACTIONS", "0") == "1")
    p.add_argument("--random-actions", default=os.environ.get("RANDOM_ACTIONS", ""), help="e.g. 0,1,2,3,4,5 or empty for default")
    p.add_argument("--random-stop-prob", type=float, default=float(os.environ.get("RANDOM_STOP_PROB", "0.05")))
    p.add_argument("--openfly-model-path", default=os.environ.get("OPENFLY_LOCAL_MODEL_PATH", "models/openfly-agent-7b"))
    p.add_argument("--openvla-model-path", default=os.environ.get("OPENVLA_LOCAL_MODEL_PATH", "models/openvla-7b-prismatic"))
    p.add_argument("--lite-ckpt", default=os.environ.get("LITE_CKPT", ""), help="checkpoint from baseline_eval_3090/lite/train_lite_baseline.py")
    p.add_argument("--unnorm-key", default=os.environ.get("UNNORM_KEY", "vln_norm"))
    p.add_argument("--fallback-unnorm-keys", default=os.environ.get("FALLBACK_UNNORM_KEYS", "vln_norm,vlnv1,openfly,bridge_orig"))
    p.add_argument("--api-base-url", default=os.environ.get("OPENAI_BASE_URL", os.environ.get("API_BASE_URL", "")))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", os.environ.get("API_KEY", "")))
    p.add_argument("--api-model", default=os.environ.get("VLM_MODEL", os.environ.get("API_MODEL", "")))
    p.add_argument("--api-temperature", type=float, default=float(os.environ.get("API_TEMPERATURE", "0")))
    p.add_argument("--api-timeout", type=int, default=int(os.environ.get("API_TIMEOUT", "120")))
    p.add_argument("--api-frames", type=int, default=int(os.environ.get("API_FRAMES", "3")))
    p.add_argument("--api-max-retries", type=int, default=int(os.environ.get("API_MAX_RETRIES", "2")))
    p.add_argument("--default-action-on-error", type=int, default=int(os.environ.get("DEFAULT_ACTION_ON_ERROR", "1")))
    return p


def make_config(args: argparse.Namespace) -> EvalConfig:
    method = args.method
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / method / stamp
    return EvalConfig(
        method=method,
        eval_info=args.eval_info,
        output_dir=output_dir,
        env_filter=parse_env_filter(args.env_filter),
        limit_total=args.limit_total,
        limit_per_env=args.limit_per_env,
        max_step=args.max_step,
        seed=args.seed,
        device=args.device,
        image_history=args.image_history,
        pos_ratio_gs=args.pos_ratio_gs,
        stop_threshold_m=args.stop_threshold_m,
        dry_run=args.dry_run,
        save_images=args.save_images,
        force_six_actions=args.force_six_actions,
        random_stop_prob=args.random_stop_prob,
        random_actions=action_list_from_env(args.force_six_actions, args.random_actions),
        openfly_model_path=args.openfly_model_path,
        openvla_model_path=args.openvla_model_path,
        lite_ckpt=args.lite_ckpt,
        unnorm_key=args.unnorm_key,
        fallback_unnorm_keys=[x.strip() for x in args.fallback_unnorm_keys.split(",") if x.strip()],
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        api_model=args.api_model,
        api_temperature=args.api_temperature,
        api_timeout=args.api_timeout,
        api_frames=args.api_frames,
        api_max_retries=args.api_max_retries,
        default_action_on_error=args.default_action_on_error,
    )


def load_eval_items(cfg: EvalConfig) -> List[Dict[str, Any]]:
    path = Path(cfg.eval_info)
    if not path.exists():
        raise FileNotFoundError(f"eval info not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"eval info must be a JSON list: {path}")

    def keep(item: Dict[str, Any]) -> bool:
        env = str(item.get("image_path", "")).split("/")[0]
        if cfg.env_filter is None:
            return True
        return any(f in env or f in item.get("image_path", "") for f in cfg.env_filter)

    data = [x for x in data if keep(x)]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        env = item["image_path"].split("/")[0]
        grouped.setdefault(env, []).append(item)
    out: List[Dict[str, Any]] = []
    for env in sorted(grouped.keys()):
        items = grouped[env]
        if cfg.limit_per_env is not None:
            items = items[: cfg.limit_per_env]
        out.extend(items)
    if cfg.limit_total is not None:
        out = out[: cfg.limit_total]
    return out


def group_by_env(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        env = item["image_path"].split("/")[0]
        groups.setdefault(env, []).append(item)
    return groups


def create_policy(cfg: EvalConfig) -> BasePolicy:
    if cfg.method == "random":
        return RandomPolicy(actions=cfg.random_actions, stop_prob=cfg.random_stop_prob, seed=cfg.seed)
    if cfg.method == "oracle_replay":
        return OracleReplayPolicy()
    if cfg.method == "openfly":
        return OpenFlyPolicy(cfg.openfly_model_path, device=cfg.device, unnorm_key=cfg.unnorm_key, fallback_keys=cfg.fallback_unnorm_keys)
    if cfg.method == "openvla_direct":
        return OpenVLADirectPolicy(cfg.openvla_model_path, device=cfg.device, unnorm_key=cfg.unnorm_key)
    if cfg.method in {"seq2seq_lite", "cma_lite", "aerialvln_lite"}:
        return LiteCheckpointPolicy(cfg.lite_ckpt, device=cfg.device)
    if cfg.method in {"spf_api", "navid_direct_api", "navila_direct_api"}:
        return APIVLMPolicy(
            variant=cfg.method,
            base_url=cfg.api_base_url,
            api_key=cfg.api_key,
            model=cfg.api_model,
            temperature=cfg.api_temperature,
            timeout=cfg.api_timeout,
            frames=cfg.api_frames,
            retries=cfg.api_max_retries,
            default_action=cfg.default_action_on_error,
        )
    raise ValueError(f"unknown method: {cfg.method}")


def create_env_bridge(env_name: str):
    if "airsim" in env_name:
        return AirsimBridge(env_name), 1.0
    if "ue" in env_name:
        return UEBridge(ue_ip="127.0.0.1", ue_port="9000", env_name=env_name), 1.0
    if "gs" in env_name:
        return GSBridge(env_name), 5.15
    raise ValueError(f"Unknown environment type: {env_name}")


def cleanup_env() -> None:
    for kw in [
        "AirVLN", "AirSim", "LinuxNoEditor", "AirVLN-Linux-Shipping", "CitySample", "CrashReport",
        "guangzhou", "shanghai", "Unreal", "UE4", "UE5", "SIBR_gaussianHierarchyViewer_app",
    ]:
        try:
            kill_env_process(kw)
        except Exception:
            pass


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0, "NE": None, "SR": None, "OSR": None, "SPL": None}
    return {
        "n": len(rows),
        "NE": float(np.mean([r["NE"] for r in rows])),
        "SR": float(np.mean([r["SR"] for r in rows])),
        "OSR": float(np.mean([r["OSR"] for r in rows])),
        "SPL": float(np.mean([r["SPL"] for r in rows])),
        "stop_rate": float(np.mean([1.0 if r.get("stopped", False) else 0.0 for r in rows])),
        "avg_steps": float(np.mean([r.get("steps", 0) for r in rows])),
    }


def write_summaries(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # CSV details
    detail_csv = output_dir / "episode_results.csv"
    fieldnames = [
        "env", "sample_idx", "global_idx", "NE", "SR", "OSR", "SPL", "steps", "stopped", "image_error",
        "start", "end", "final_pose", "actions", "instruction",
    ]
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            for k in ["start", "end", "final_pose", "actions"]:
                rr[k] = json.dumps(rr.get(k), ensure_ascii=False)
            w.writerow({k: rr.get(k) for k in fieldnames})

    # Summary by env and overall
    by_env: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_env.setdefault(r["env"], []).append(r)
    summary = {"overall": aggregate(rows), "by_env": {env: aggregate(rs) for env, rs in sorted(by_env.items())}}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "n", "NE", "SR", "OSR", "SPL", "stop_rate", "avg_steps"])
        for scope, item in [("overall", summary["overall"])] + [(k, v) for k, v in summary["by_env"].items()]:
            w.writerow([scope, item.get("n"), item.get("NE"), item.get("SR"), item.get("OSR"), item.get("SPL"), item.get("stop_rate"), item.get("avg_steps")])


def run_eval(cfg: EvalConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "config.json").write_text(json.dumps(cfg.__dict__ | {"output_dir": str(cfg.output_dir)}, ensure_ascii=False, indent=2), encoding="utf-8")

    items = load_eval_items(cfg)
    groups = group_by_env(items)
    print("=" * 100)
    print(f"[OpenFly Baseline Eval 3090] method={cfg.method} eval_info={cfg.eval_info}")
    print(f"[Output] {cfg.output_dir}")
    print(f"[Items] total={len(items)} groups={ {k: len(v) for k, v in groups.items()} }")
    print("=" * 100)
    if cfg.dry_run:
        return

    policy = create_policy(cfg)
    rows: List[Dict[str, Any]] = []
    jsonl_path = cfg.output_dir / "episode_results.jsonl"
    images_dir = cfg.output_dir / "images"
    if cfg.save_images:
        images_dir.mkdir(exist_ok=True)

    global_idx = 0
    try:
        for env_name, env_items in groups.items():
            print(f"\n[ENV] Starting {env_name} with {len(env_items)} episodes")
            time.sleep(3)
            env_bridge = None
            try:
                env_bridge, pos_ratio = create_env_bridge(env_name)
                if "gs" in env_name:
                    pos_ratio = cfg.pos_ratio_gs
                for sample_idx, item in enumerate(env_items):
                    global_idx += 1
                    pos_list = item["pos"]
                    instruction = item["gpt_instruction"]
                    start_position = pos_list[0]
                    end_position = pos_list[-1]
                    start_yaw = item["yaw"][0]
                    new_pose = [start_position[0], start_position[1], start_position[2], start_yaw]
                    old_pose = list(new_pose)
                    env_bridge.pass_len = 1e-3
                    image_history: List[np.ndarray] = []
                    actions: List[int] = []
                    stopped = False
                    image_error = False
                    flag_osr = 0
                    policy.reset(instruction, item)
                    pitch = -45.0 if "high" in item.get("image_path", "") else 0.0
                    print(f"\n[EP] env={env_name} sample={sample_idx} global={global_idx} start={start_position} end={end_position} yaw={start_yaw}")
                    env_bridge.set_camera_pose(start_position[0] / pos_ratio, start_position[1] / pos_ratio, start_position[2] / pos_ratio, pitch, np.rad2deg(start_yaw), 0)
                    step = 0
                    while step < cfg.max_step:
                        try:
                            raw_image = env_bridge.get_camera_data()
                            if raw_image is None:
                                raise RuntimeError("env returned None image")
                            image_history.append(raw_image)
                            if len(image_history) > max(cfg.image_history, 3):
                                image_history = image_history[-max(cfg.image_history, 3):]
                            if cfg.save_images:
                                cv2.imwrite(str(images_dir / f"{env_name}_{global_idx:05d}_{step:03d}.jpg"), raw_image)

                            action_id = int(policy.act(image_history, instruction, item, step, new_pose))
                            if cfg.force_six_actions and action_id > 5:
                                # Collapse long/side actions into six-action UAV space.
                                action_id = 1 if action_id in {8, 9, 6, 7} else action_id
                            action_id = max(0, min(9, action_id))
                            actions.append(action_id)
                            print(f"[STEP] env={env_name} ep={sample_idx} step={step} action={action_id}:{ACTION_ID_TO_NAME.get(action_id)} pose_before={new_pose}")

                            new_pose = getPoseAfterMakeAction(new_pose, action_id)
                            env_bridge.set_camera_pose(new_pose[0] / pos_ratio, new_pose[1] / pos_ratio, new_pose[2] / pos_ratio, pitch, np.rad2deg(new_pose[3]), 0)
                            env_bridge.pass_len += calculate_distance(old_pose, new_pose)
                            old_pose = list(new_pose)
                            dis_now = calculate_distance(end_position, new_pose)
                            if dis_now < cfg.stop_threshold_m and flag_osr == 0:
                                flag_osr = 1
                            if action_id == 0:
                                stopped = True
                                break
                            step += 1
                        except Exception as e:
                            image_error = True
                            print(f"[ERROR] step failed: {repr(e)}")
                            traceback.print_exc()
                            break

                    dis = calculate_distance(end_position, new_pose)
                    traj_len = calculate_distance(end_position, start_position)
                    sr = 1 if dis < cfg.stop_threshold_m else 0
                    osr = 1 if flag_osr else 0
                    spl = float(traj_len / max(env_bridge.pass_len, 1e-6)) if sr else 0.0
                    row = {
                        "env": env_name,
                        "sample_idx": sample_idx,
                        "global_idx": global_idx,
                        "NE": float(dis),
                        "SR": int(sr),
                        "OSR": int(osr),
                        "SPL": float(spl),
                        "steps": len(actions),
                        "stopped": bool(stopped),
                        "image_error": bool(image_error),
                        "start": start_position,
                        "end": end_position,
                        "final_pose": new_pose,
                        "actions": actions,
                        "instruction": instruction,
                    }
                    rows.append(row)
                    with jsonl_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"[EP RESULT] NE={dis:.3f} SR={sr} OSR={osr} SPL={spl:.4f} steps={len(actions)} stopped={stopped}")
                    write_summaries(cfg.output_dir, rows)
            finally:
                print(f"[ENV] Cleaning up {env_name}")
                try:
                    del env_bridge
                except Exception:
                    pass
                cleanup_env()
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    finally:
        policy.close()
        write_summaries(cfg.output_dir, rows)
        print("\n" + "=" * 100)
        print("[DONE] Final summary:")
        print(json.dumps(aggregate(rows), ensure_ascii=False, indent=2))
        print(f"[FILES] {cfg.output_dir}/summary.csv  {cfg.output_dir}/episode_results.csv")
        print("=" * 100)


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = make_config(args)
    run_eval(cfg)


if __name__ == "__main__":
    main()
