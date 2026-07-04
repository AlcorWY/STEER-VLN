import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class WaypointDecoderConfig:
    d_scale: float = 12.0
    z_scale: float = 5.0

    yaw_turn_threshold: float = 0.35
    yaw_strong_threshold: float = 0.75

    use_z_decode: bool = False
    z_threshold: float = 1.0

    forward_policy: str = "single_step"
    forward_3_threshold: float = 4.5
    forward_6_threshold: float = 8.0

    # STEER-VLN final: stop 由 StopTrendStateModule 决定，这里默认关闭旧 action_logits stop。
    use_aux_action_stop: bool = False
    stop_prob_threshold: float = 0.50
    stop_logit_margin: float = 0.0

    stop_id: int = 0
    forward3_id: int = 1
    turn_left_id: int = 2
    turn_right_id: int = 3
    up_id: int = 4
    down_id: int = 5
    move_left_id: int = 6
    move_right_id: int = 7
    forward6_id: int = 8
    forward9_id: int = 9


def denormalize_waypoint(
    waypoint_pred: torch.Tensor,
    cfg: Optional[WaypointDecoderConfig] = None,
) -> torch.Tensor:
    if cfg is None:
        cfg = WaypointDecoderConfig()

    out = waypoint_pred.detach().float().clone()
    out[..., 0] = out[..., 0].clamp(min=0.0) * cfg.d_scale
    out[..., 1] = out[..., 1].clamp(min=-1.5, max=1.5) * math.pi
    out[..., 2] = out[..., 2].clamp(min=-1.5, max=1.5) * cfg.z_scale
    return out


def decode_one_waypoint(
    d_m: float,
    yaw_rad: float,
    z_m: float,
    cfg: Optional[WaypointDecoderConfig] = None,
) -> int:
    if cfg is None:
        cfg = WaypointDecoderConfig()

    if cfg.use_z_decode:
        if z_m > cfg.z_threshold:
            return cfg.up_id
        if z_m < -cfg.z_threshold:
            return cfg.down_id

    if yaw_rad > cfg.yaw_turn_threshold:
        return cfg.turn_left_id

    if yaw_rad < -cfg.yaw_turn_threshold:
        return cfg.turn_right_id

    if cfg.forward_policy == "single_step":
        return cfg.forward3_id

    if cfg.forward_policy == "distance_bins":
        if d_m <= cfg.forward_3_threshold:
            return cfg.forward3_id
        if d_m <= cfg.forward_6_threshold:
            return cfg.forward6_id
        return cfg.forward9_id

    raise ValueError(f"Unknown forward_policy: {cfg.forward_policy}")


def decode_waypoint_batch(
    waypoint_pred: torch.Tensor,
    cfg: Optional[WaypointDecoderConfig] = None,
) -> torch.Tensor:
    if cfg is None:
        cfg = WaypointDecoderConfig()

    wp = denormalize_waypoint(waypoint_pred, cfg)
    actions = []

    for i in range(wp.shape[0]):
        actions.append(
            decode_one_waypoint(
                float(wp[i, 0].item()),
                float(wp[i, 1].item()),
                float(wp[i, 2].item()),
                cfg,
            )
        )

    return torch.tensor(actions, dtype=torch.long, device=waypoint_pred.device)


def aux_stop_mask(
    action_logits: torch.Tensor,
    cfg: Optional[WaypointDecoderConfig] = None,
) -> torch.Tensor:
    """
    兼容旧消融用。STEER_VLN 最终方案默认 use_aux_action_stop=False。
    """
    if cfg is None:
        cfg = WaypointDecoderConfig()

    if not cfg.use_aux_action_stop:
        return torch.zeros(
            action_logits.shape[0],
            dtype=torch.bool,
            device=action_logits.device,
        )

    logits = action_logits.float()
    probs = torch.softmax(logits, dim=-1)

    stop_prob = probs[:, cfg.stop_id]
    stop_logit = logits[:, cfg.stop_id]

    non_stop_logits = logits.clone()
    non_stop_logits[:, cfg.stop_id] = -1e9
    best_non_stop = non_stop_logits.max(dim=-1).values

    by_prob = stop_prob >= cfg.stop_prob_threshold
    by_margin = stop_logit >= best_non_stop + cfg.stop_logit_margin

    return by_prob | by_margin


def hybrid_aux_stop_decode_batch(
    action_logits: torch.Tensor,
    waypoint_pred: torch.Tensor,
    cfg: Optional[WaypointDecoderConfig] = None,
) -> torch.Tensor:
    """
    兼容旧版 hybrid decoder。
    STEER_VLN 最终方案中 stop 由 StopTrendStateModule/runtime 覆盖，不由这里决定。
    """
    if cfg is None:
        cfg = WaypointDecoderConfig()

    wp_actions = decode_waypoint_batch(waypoint_pred, cfg)
    stop_mask = aux_stop_mask(action_logits, cfg)

    final_actions = wp_actions.clone()
    final_actions[stop_mask] = cfg.stop_id

    return final_actions


@torch.no_grad()
def decoder_metrics(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    cfg: Optional[WaypointDecoderConfig] = None,
) -> Dict[str, float]:
    if cfg is None:
        cfg = WaypointDecoderConfig()

    pred = pred_actions.detach().cpu()
    gt = gt_actions.detach().cpu()

    acc = (pred == gt).float().mean().item()

    stop_mask = gt == cfg.stop_id
    turn_mask = (gt == cfg.turn_left_id) | (gt == cfg.turn_right_id)
    forward_mask = (
        (gt == cfg.forward3_id)
        | (gt == cfg.forward6_id)
        | (gt == cfg.forward9_id)
    )

    out = {
        "decoder_action_acc": float(acc),
        "pred_stop_rate": float((pred == cfg.stop_id).float().mean().item()),
        "gt_stop_rate": float((gt == cfg.stop_id).float().mean().item()),
        "pred_turn_rate": float(
            ((pred == cfg.turn_left_id) | (pred == cfg.turn_right_id)).float().mean().item()
        ),
        "gt_turn_rate": float(turn_mask.float().mean().item()),
    }

    if stop_mask.sum().item() > 0:
        out["stop_action_acc"] = float((pred[stop_mask] == gt[stop_mask]).float().mean().item())
        out["stop_count"] = int(stop_mask.sum().item())
    else:
        out["stop_action_acc"] = 0.0
        out["stop_count"] = 0

    if turn_mask.sum().item() > 0:
        out["turn_action_acc"] = float((pred[turn_mask] == gt[turn_mask]).float().mean().item())
        out["turn_count"] = int(turn_mask.sum().item())
    else:
        out["turn_action_acc"] = 0.0
        out["turn_count"] = 0

    if forward_mask.sum().item() > 0:
        out["forward_action_acc"] = float((pred[forward_mask] == gt[forward_mask]).float().mean().item())
        out["forward_count"] = int(forward_mask.sum().item())
    else:
        out["forward_action_acc"] = 0.0
        out["forward_count"] = 0

    return out
