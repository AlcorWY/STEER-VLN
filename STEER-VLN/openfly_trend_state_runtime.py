# ===== STEER_VLN trend-state runtime import guard =====
import sys as _steer_vln_sys
from pathlib import Path as _STEER_VLNPath

_STEER_VLN_FILE = _STEER_VLNPath(__file__).resolve()
_STEER_VLN_DIR = _STEER_VLN_FILE.parent
_STEER_VLN_ROOT = _STEER_VLN_DIR.parents[0]
_STEER_VLN_CODE = _STEER_VLN_ROOT / "code"
_STEER_VLN_TRAIN = _STEER_VLN_ROOT / "train"

for _p in ["", str(_STEER_VLN_ROOT), str(_STEER_VLN_CODE), str(_STEER_VLN_DIR), str(_STEER_VLN_TRAIN)]:
    if _p in _steer_vln_sys.path:
        _steer_vln_sys.path.remove(_p)

_steer_vln_sys.path.insert(0, str(_STEER_VLN_ROOT))
_steer_vln_sys.path.insert(0, str(_STEER_VLN_CODE))
_steer_vln_sys.path.insert(0, str(_STEER_VLN_DIR))
_steer_vln_sys.path.append(str(_STEER_VLN_TRAIN))
# ===== end STEER_VLN trend-state runtime import guard =====

from typing import Dict, Sequence, List

import numpy as np
import torch
from PIL import Image

from openfly_feature_trend_head import (
    build_progress_from_step,
    OpenFlyFeatureTrendHead,
    OpenFlyFeatureTrendHeadConfig,
)
from keyframe.waypoint_decoder import (
    WaypointDecoderConfig,
    decode_waypoint_batch,
    denormalize_waypoint,
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


def frame_bundle_to_pil(frame_bundle):
    images = []
    for frame in frame_bundle:
        if isinstance(frame, Image.Image):
            images.append(frame.convert("RGB"))
            continue
        arr = frame
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


def move_processor_inputs(inputs, device, dtype):
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if v.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def _get_by_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_language_final_norm(policy):
    """
    Same feature extraction strategy as train_integrated_full.py.
    Hook the language model final norm and avoid output_hidden_states=True,
    which keeps all layer hidden states in memory.
    """
    candidates = [
        "language_model.base_model.model.model.norm",
        "language_model.model.model.norm",
        "language_model.model.norm",
        "language_model.base_model.model.model.final_layernorm",
        "language_model.model.model.final_layernorm",
    ]

    for path in candidates:
        mod = _get_by_path(policy, path)
        if mod is not None:
            return mod, path

    return None, None


def _label_distance(d: float) -> str:
    d = abs(float(d))
    if d <= 4.5:
        return "near"
    if d <= 8.0:
        return "middle"
    return "far"


def _label_yaw(yaw_rad: float, th: float = 0.35) -> str:
    yaw_rad = float(yaw_rad)
    if yaw_rad > th:
        return "left"
    if yaw_rad < -th:
        return "right"
    return "straight"


def _label_z(z_m: float, th: float = 1.0) -> str:
    z_m = float(z_m)
    if z_m > th:
        return "up"
    if z_m < -th:
        return "down"
    return "stable"


class OpenFlyTrendStateRuntime:
    """
    Final STEER-VLN runtime.

    This module only provides auxiliary trend state.
    It never overrides OpenFly/LoRA final action.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        d_scale: float = 12.0,
        z_scale: float = 5.0,
        yaw_turn_threshold: float = 0.35,
        z_threshold: float = 1.0,
        use_z_decode: bool = True,
        forward_policy: str = "distance_bins",
        forward_3_threshold: float = 4.5,
        forward_6_threshold: float = 8.0,
        stop_threshold: float = 0.50,
        prestop_threshold: float = 0.50,
        temporal_window: int = 3,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg_dict = dict(ckpt["cfg"])
        cfg_dict.setdefault("use_stop_trend_state", True)
        cfg_dict.setdefault("use_temporal_stop_state", True)
        cfg_dict.setdefault("stop_attention_heads", 4)
        cfg_dict.setdefault("stop_attention_dropout", cfg_dict.get("dropout", 0.1))
        cfg_dict.setdefault("stop_context_detach", True)
        cfg_dict.setdefault("temporal_stop_window", int(temporal_window))
        cfg_dict.setdefault("temporal_stop_heads", 4)
        cfg_dict.setdefault("temporal_stop_dropout", cfg_dict.get("dropout", 0.1))

        cfg = OpenFlyFeatureTrendHeadConfig(**cfg_dict)
        self.head = OpenFlyFeatureTrendHead(cfg)

        missing, unexpected = self.head.load_state_dict(ckpt["head"], strict=False)
        if missing:
            print(f"[TrendStateRuntime][WARN] missing keys: {missing[:10]} total={len(missing)}", flush=True)
        if unexpected:
            print(f"[TrendStateRuntime][WARN] unexpected keys: {unexpected[:10]} total={len(unexpected)}", flush=True)

        self.head.to(self.device)
        self.head.eval()

        self.decoder_cfg = WaypointDecoderConfig(
            d_scale=d_scale,
            z_scale=z_scale,
            yaw_turn_threshold=yaw_turn_threshold,
            z_threshold=z_threshold,
            use_z_decode=use_z_decode,
            forward_policy=forward_policy,
            forward_3_threshold=forward_3_threshold,
            forward_6_threshold=forward_6_threshold,
        )

        self.stop_threshold = float(stop_threshold)
        self.prestop_threshold = float(prestop_threshold)
        self.temporal_window = int(temporal_window)

        self.stop_token_history: List[torch.Tensor] = []
        self.stop_prob_history: List[float] = []
        self.prestop_prob_history: List[float] = []

        print("[OpenFlyTrendStateRuntime] loaded", flush=True)
        print(f"  checkpoint: {checkpoint_path}", flush=True)
        print(f"  device: {self.device}", flush=True)
        print(f"  decoder_cfg: {self.decoder_cfg}", flush=True)

    def reset_episode(self):
        self.stop_token_history = []
        self.stop_prob_history = []
        self.prestop_prob_history = []

    def _build_stop_history_tensor(self):
        if not self.stop_token_history:
            return None
        hist = self.stop_token_history[-self.temporal_window:]
        return torch.stack(hist, dim=1).to(self.device)

    def _append_stop_token(self, token: torch.Tensor):
        self.stop_token_history.append(token.detach())
        self.stop_token_history = self.stop_token_history[-self.temporal_window:]

    @torch.no_grad()
    def extract_feature(self, policy, processor, frame_bundle: Sequence, text: str):
        images = frame_bundle_to_pil(frame_bundle)
        inputs = processor(str(text).lower(), images, return_tensors="pt")
        inputs = move_processor_inputs(inputs, self.device, self.dtype)

        norm_module, norm_path = find_language_final_norm(policy)

        if norm_module is None:
            # Keep a fallback only for unexpected model layouts.
            print(
                "[TrendStateRuntime][WARN] final norm hook not found; "
                "fallback to output_hidden_states=True",
                flush=True,
            )
            outputs = policy(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                raise RuntimeError("policy output has no hidden_states.")

            last_hidden = outputs.hidden_states[-1]
        else:
            captured = {}

            def hook_fn(module, hook_inputs, hook_output):
                captured["hidden"] = hook_output

            handle = norm_module.register_forward_hook(hook_fn)
            try:
                # Match train_integrated_full.py: do not materialize all hidden states.
                _ = policy(
                    **inputs,
                    output_hidden_states=False,
                    return_dict=True,
                )
            finally:
                handle.remove()

            if "hidden" not in captured:
                raise RuntimeError(f"final norm hook did not capture hidden state at {norm_path}")

            last_hidden = captured["hidden"]

        if "attention_mask" in inputs:
            attn = inputs["attention_mask"]
            idx = attn.sum(dim=1).long() - 1
            feat = last_hidden[torch.arange(last_hidden.shape[0], device=self.device), idx]
        else:
            feat = last_hidden[:, -1]

        return feat.float()

    def _temporal_label(self, stop_prob: float, prestop_prob: float) -> str:
        self.stop_prob_history.append(float(stop_prob))
        self.prestop_prob_history.append(float(prestop_prob))

        self.stop_prob_history = self.stop_prob_history[-self.temporal_window:]
        self.prestop_prob_history = self.prestop_prob_history[-self.temporal_window:]

        stop_hits = sum(x >= self.stop_threshold for x in self.stop_prob_history)
        prestop_hits = sum(x >= self.prestop_threshold for x in self.prestop_prob_history)

        if len(self.stop_prob_history) >= self.temporal_window:
            if stop_hits >= self.temporal_window and prestop_hits >= self.temporal_window - 1:
                return "stable_stop_candidate"
            if prestop_hits >= self.temporal_window - 1:
                return "near_target_but_continue_checking"

        if prestop_prob >= self.prestop_threshold:
            return "near_target"
        return "continue_moving"

    @torch.no_grad()
    def predict_state(self, policy, processor, frame_bundle: Sequence, text: str, step: int = 0, max_step: int = 100) -> Dict[str, object]:
        feature = self.extract_feature(
            policy=policy,
            processor=processor,
            frame_bundle=frame_bundle,
            text=text,
        )

        progress = build_progress_from_step(step, max_step, feature.device)
        stop_history = self._build_stop_history_tensor()

        out = self.head(
            feature,
            progress=progress,
            stop_history=stop_history,
        )

        action_logits = out["action_logits"]
        waypoint_pred = out["waypoint_pred"]

        single_stop_prob = float(torch.sigmoid(out.get("single_stop_logit", out["stop_logit"]))[0].item())
        temporal_stop_prob = float(torch.sigmoid(out["stop_logit"])[0].item())
        prestop_prob = float(torch.sigmoid(out["prestop_logit"])[0].item())

        action_probs = torch.softmax(action_logits.float(), dim=-1)[0]
        aux_action = int(action_probs.argmax().item())
        aux_action_name = ACTION_NAMES.get(aux_action, f"action_{aux_action}")

        decoder_action = int(decode_waypoint_batch(waypoint_pred, self.decoder_cfg)[0].item())
        decoder_action_name = ACTION_NAMES.get(decoder_action, f"action_{decoder_action}")

        wp_raw_tensor = denormalize_waypoint(waypoint_pred, self.decoder_cfg)[0]
        d_m = float(wp_raw_tensor[0].item())
        yaw_rad = float(wp_raw_tensor[1].item())
        z_m = float(wp_raw_tensor[2].item())

        distance_trend = _label_distance(d_m)
        yaw_trend = _label_yaw(yaw_rad, self.decoder_cfg.yaw_turn_threshold)
        z_trend = _label_z(z_m, self.decoder_cfg.z_threshold)

        temporal_stop_trend = self._temporal_label(temporal_stop_prob, prestop_prob)

        if "stop_token" in out:
            self._append_stop_token(out["stop_token"])

        trend_text = (
            " Auxiliary navigation trend state: "
            f"distance={distance_trend}; "
            f"yaw={yaw_trend}; "
            f"vertical={z_trend}; "
            f"motion_hint={aux_action_name}; "
            f"waypoint_hint={decoder_action_name}; "
            f"single_stop_probability={single_stop_prob:.2f}; "
            f"temporal_stop_probability={temporal_stop_prob:.2f}; "
            f"prestop_probability={prestop_prob:.2f}; "
            f"temporal_stop_trend={temporal_stop_trend}. "
            "Use these trend states only as auxiliary context. "
            "Preserve the full movement action space and decide the final action yourself."
        )

        return {
            "conditioned_text": str(text).strip() + trend_text,
            "trend_text": trend_text,
            "aux_action": aux_action,
            "aux_action_name": aux_action_name,
            "decoder_action": decoder_action,
            "decoder_action_name": decoder_action_name,
            "waypoint_pred_raw": [float(x) for x in wp_raw_tensor.detach().cpu().tolist()],
            "distance_trend": distance_trend,
            "yaw_trend": yaw_trend,
            "vertical_trend": z_trend,
            "single_stop_prob": single_stop_prob,
            "stop_prob": temporal_stop_prob,
            "prestop_prob": prestop_prob,
            "temporal_stop_trend": temporal_stop_trend,
            "stop_history_len": len(self.stop_token_history),
        }
