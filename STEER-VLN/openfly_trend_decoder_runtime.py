# ===== STEER_VLN runtime import guard =====
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
# ===== end STEER_VLN runtime import guard =====

from typing import Dict, Optional, Sequence

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


class OpenFlyTrendDecoderRuntime:
    """
    M4-S runtime.

    Reuse already-loaded OpenFly-Agent policy and processor:
      [keyframe, previous, current]
        -> OpenFly hidden feature
        -> OpenFlyFeatureTrendHead
        -> waypoint decoder + stop/prestop trend gate
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
        forward_policy: str = "single_step",
        forward_3_threshold: float = 4.5,
        forward_6_threshold: float = 8.0,
        stop_prob_threshold: float = 0.5,
        stop_logit_margin: float = 0.0,
        trend_stop_threshold: float = 0.50,
        trend_prestop_threshold: float = 0.20,
        trend_use_margin: bool = False,
        trend_margin_threshold: float = -20.0,
        trend_use_distance: bool = False,
        trend_distance_threshold: float = 999.0,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = OpenFlyFeatureTrendHeadConfig(**ckpt["cfg"])

        self.head = OpenFlyFeatureTrendHead(cfg)
        self.head.load_state_dict(ckpt["head"], strict=True)
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

        # Legacy arguments kept for compatibility with older waypoint decoder eval code.
        # M4-S / E5S-HD uses trend_stop_threshold + trend_prestop_threshold instead.
        self.legacy_stop_prob_threshold = float(stop_prob_threshold)
        self.legacy_stop_logit_margin = float(stop_logit_margin)

        self.trend_stop_threshold = float(trend_stop_threshold)
        self.trend_prestop_threshold = float(trend_prestop_threshold)
        self.trend_use_margin = bool(trend_use_margin)
        self.trend_margin_threshold = float(trend_margin_threshold)
        self.trend_use_distance = bool(trend_use_distance)
        self.trend_distance_threshold = float(trend_distance_threshold)

        print("[OpenFlyTrendDecoderRuntime] loaded", flush=True)
        print(f"  checkpoint: {checkpoint_path}", flush=True)
        print(f"  device: {self.device}", flush=True)
        print(f"  decoder: {self.decoder_cfg}", flush=True)
        print(
            "  trend gate: "
            f"stop>={self.trend_stop_threshold}, "
            f"prestop>={self.trend_prestop_threshold}, "
            f"use_margin={self.trend_use_margin}, "
            f"margin>={self.trend_margin_threshold}, "
            f"use_distance={self.trend_use_distance}, "
            f"distance<={self.trend_distance_threshold}",
            flush=True,
        )

    @torch.no_grad()
    def extract_feature(self, policy, processor, frame_bundle: Sequence, text: str):
        images = frame_bundle_to_pil(frame_bundle)

        inputs = processor(
            str(text).lower(),
            images,
            return_tensors="pt",
        )
        inputs = move_processor_inputs(inputs, self.device, self.dtype)

        outputs = policy(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError(
                "policy output has no hidden_states. "
                "Need output_hidden_states=True support or a forward hook."
            )

        last_hidden = outputs.hidden_states[-1]

        if "attention_mask" in inputs:
            attn = inputs["attention_mask"]
            idx = attn.sum(dim=1).long() - 1
            feat = last_hidden[torch.arange(last_hidden.shape[0], device=self.device), idx]
        else:
            feat = last_hidden[:, -1]

        return feat.float()

    @torch.no_grad()
    def predict(self, policy, processor, frame_bundle: Sequence, text: str, step: int = 0, max_step: int = 100) -> Dict[str, object]:
        feature = self.extract_feature(
            policy=policy,
            processor=processor,
            frame_bundle=frame_bundle,
            text=text,
        )

        progress = build_progress_from_step(step, max_step, feature.device)
        out = self.head(feature, progress=progress)

        action_logits = out["action_logits"]
        waypoint_pred = out["waypoint_pred"]
        stop_prob = torch.sigmoid(out["stop_logit"])[0]
        prestop_prob = torch.sigmoid(out["prestop_logit"])[0]

        aux_action = int(action_logits.argmax(dim=-1)[0].item())
        decoder_action = int(decode_waypoint_batch(waypoint_pred, self.decoder_cfg)[0].item())

        wp_raw_tensor = denormalize_waypoint(waypoint_pred, self.decoder_cfg)[0]
        wp_raw = wp_raw_tensor.detach().cpu().tolist()
        wp_distance = float(abs(wp_raw_tensor[0].item()))

        logits = action_logits.float()
        stop_action_logit = logits[:, self.decoder_cfg.stop_id]
        non_stop_max = logits[:, 1:].max(dim=1).values
        stop_action_margin = float((stop_action_logit - non_stop_max)[0].item())

        trend_mask = (
            float(stop_prob.item()) >= self.trend_stop_threshold
            and float(prestop_prob.item()) >= self.trend_prestop_threshold
        )

        if self.trend_use_margin:
            trend_mask = trend_mask and (stop_action_margin >= self.trend_margin_threshold)

        if self.trend_use_distance:
            trend_mask = trend_mask and (wp_distance <= self.trend_distance_threshold)

        trend_hybrid_action = self.decoder_cfg.stop_id if trend_mask else decoder_action

        probs = torch.softmax(action_logits.float(), dim=-1)[0].detach().cpu().tolist()

        return {
            "aux_action": aux_action,
            "decoder_action": decoder_action,
            "hybrid_aux_action": trend_hybrid_action,
            "trend_hybrid_action": trend_hybrid_action,
            "waypoint_pred_norm": waypoint_pred[0].detach().cpu().tolist(),
            "waypoint_pred_raw": wp_raw,
            "action_probs": probs,
            "stop_prob": float(stop_prob.item()),
            "prestop_prob": float(prestop_prob.item()),
            "stop_action_margin": stop_action_margin,
            "wp_distance": wp_distance,
            "trend_stop_mask": bool(trend_mask),
        }


def select_final_action(
    action_mode: str,
    openfly_action: int,
    waypoint_result: Optional[Dict[str, object]],
) -> int:
    mode = str(action_mode).lower()

    if mode == "openfly":
        return int(openfly_action)

    if waypoint_result is None:
        return int(openfly_action)

    if mode == "aux_action":
        return int(waypoint_result["aux_action"])

    if mode == "waypoint_decoder":
        return int(waypoint_result["decoder_action"])

    if mode in ["m4s_trend_hybrid", "trend_hybrid", "stop_aware_hybrid"]:
        return int(waypoint_result["trend_hybrid_action"])

    if mode == "hybrid_openfly_stop":
        if int(openfly_action) == 0:
            return 0
        return int(waypoint_result["decoder_action"])

    raise ValueError(f"Unknown ACTION_MODE={action_mode}")

