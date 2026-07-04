import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OpenFlyFeatureTrendHeadConfig:
    input_dim: int
    hidden_dim: int = 512
    num_actions: int = 10
    dropout: float = 0.1

    # Final STEER-VLN: stop is temporal trend state.
    use_stop_trend_state: bool = True
    use_temporal_stop_state: bool = True
    stop_attention_heads: int = 4
    stop_attention_dropout: float = 0.1
    stop_context_detach: bool = True

    temporal_stop_window: int = 3
    temporal_stop_heads: int = 4
    temporal_stop_dropout: float = 0.1


class StopTrendStateModule(nn.Module):
    """
    Single-step stop trend state.

    It builds a stop-context token from:
      shared hidden
      action hidden
      waypoint hidden
      prestop probability
      progress

    This token is consumed by the temporal stop trend module.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        detach_context: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.detach_context = bool(detach_context)

        self.stop_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.base_stop_head = nn.Linear(hidden_dim, 1)
        self.prestop_head = nn.Linear(hidden_dim, 1)

        self.stop_state_proj = nn.Linear(hidden_dim, hidden_dim)
        self.shared_context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.action_context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.waypoint_context_proj = nn.Linear(hidden_dim, hidden_dim)

        self.prestop_token_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.progress_token_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.stop_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.stop_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )

    def forward(
        self,
        *,
        shared_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        waypoint_hidden: torch.Tensor,
        progress: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        bsz = shared_hidden.shape[0]
        device = shared_hidden.device
        dtype = shared_hidden.dtype

        if self.detach_context:
            action_hidden_ctx = action_hidden.detach()
            waypoint_hidden_ctx = waypoint_hidden.detach()
        else:
            action_hidden_ctx = action_hidden
            waypoint_hidden_ctx = waypoint_hidden

        stop_hidden = self.stop_branch(shared_hidden)

        base_stop_logit = self.base_stop_head(stop_hidden).squeeze(-1)
        prestop_logit = self.prestop_head(stop_hidden).squeeze(-1)

        if progress is None:
            progress = torch.zeros(bsz, 1, device=device, dtype=dtype)
        else:
            progress = progress.to(device=device, dtype=dtype).view(bsz, 1).clamp(0.0, 1.0)

        prestop_prob = torch.sigmoid(prestop_logit).view(bsz, 1).to(dtype=dtype)

        tokens = torch.stack(
            [
                self.stop_state_proj(stop_hidden),
                self.shared_context_proj(shared_hidden),
                self.action_context_proj(action_hidden_ctx),
                self.waypoint_context_proj(waypoint_hidden_ctx),
                self.prestop_token_mlp(prestop_prob),
                self.progress_token_mlp(progress),
            ],
            dim=1,
        )

        query = self.stop_query.to(device=device, dtype=dtype).expand(bsz, -1, -1)

        stop_context, stop_attention = self.stop_attention(
            query=query,
            key=tokens,
            value=tokens,
            need_weights=True,
        )

        stop_context = self.norm(stop_context.squeeze(1))

        stop_delta = self.delta_head(stop_context).squeeze(-1)
        stop_confidence = torch.sigmoid(self.confidence_head(stop_context).squeeze(-1))

        single_stop_logit = base_stop_logit + stop_confidence * stop_delta

        return {
            "single_stop_logit": single_stop_logit,
            "base_stop_logit": base_stop_logit,
            "prestop_logit": prestop_logit,
            "stop_delta": stop_delta,
            "stop_confidence": stop_confidence,
            "stop_attention": stop_attention,
            "stop_context": stop_context,
        }


class TemporalStopTrendStateModule(nn.Module):
    """
    Temporal stop trend state.

    stop_logit:
        temporal stop / commit-level trend

    single_stop_logit:
        current-frame stop trend

    prestop_logit:
        near-terminal trend

    This module provides auxiliary trend state. It does not execute actions.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        detach_context: bool = True,
        temporal_window: int = 3,
        temporal_heads: int = 4,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.temporal_window = int(temporal_window)

        self.single_step_stop = StopTrendStateModule(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            detach_context=detach_context,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(temporal_heads),
            dim_feedforward=hidden_dim * 4,
            dropout=float(temporal_dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        self.temporal_stop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        shared_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        waypoint_hidden: torch.Tensor,
        progress: torch.Tensor = None,
        stop_history: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        single = self.single_step_stop(
            shared_hidden=shared_hidden,
            action_hidden=action_hidden,
            waypoint_hidden=waypoint_hidden,
            progress=progress,
        )

        current_token = single["stop_context"]

        if stop_history is None or stop_history.numel() == 0:
            seq = current_token.unsqueeze(1)
        else:
            seq = torch.cat([stop_history, current_token.unsqueeze(1)], dim=1)

        if seq.shape[1] > self.temporal_window:
            seq = seq[:, -self.temporal_window:, :]

        temporal_seq = self.temporal_encoder(seq)
        temporal_state = self.temporal_norm(temporal_seq[:, -1, :])

        single_stop_prob = torch.sigmoid(single["single_stop_logit"]).view(-1, 1)
        prestop_prob = torch.sigmoid(single["prestop_logit"]).view(-1, 1)

        if progress is None:
            progress_feat = torch.zeros_like(single_stop_prob)
        else:
            progress_feat = progress.view(-1, 1).to(
                device=single_stop_prob.device,
                dtype=single_stop_prob.dtype,
            )

        temporal_input = torch.cat(
            [
                temporal_state,
                current_token,
                single_stop_prob,
                prestop_prob,
                progress_feat,
            ],
            dim=-1,
        )

        temporal_stop_logit = self.temporal_stop_head(temporal_input).squeeze(-1)

        return {
            "stop_logit": temporal_stop_logit,
            "single_stop_logit": single["single_stop_logit"],
            "base_stop_logit": single["base_stop_logit"],
            "prestop_logit": single["prestop_logit"],
            "stop_delta": single["stop_delta"],
            "stop_confidence": single["stop_confidence"],
            "stop_attention": single["stop_attention"],
            "stop_token": current_token,
            "temporal_stop_state": temporal_state,
            "temporal_stop_len": torch.full(
                (shared_hidden.shape[0],),
                int(seq.shape[1]),
                dtype=torch.long,
                device=shared_hidden.device,
            ),
        }


class OpenFlyFeatureTrendHead(nn.Module):
    """
    Final STEER-VLN temporal trend head.

    It predicts auxiliary trend states:
      action_logits       discrete action trend
      waypoint_pred       geometric waypoint trend
      single_stop_logit   current-frame stop trend
      prestop_logit       near-terminal trend
      stop_logit          temporal stop / commit-level trend

    The final action is generated by OpenFly/LoRA, not by this head.
    """

    def __init__(self, cfg: OpenFlyFeatureTrendHeadConfig):
        super().__init__()
        self.cfg = cfg

        self.trunk = nn.Sequential(
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.action_branch = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.waypoint_branch = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.action_head = nn.Linear(cfg.hidden_dim, cfg.num_actions)
        self.waypoint_head = nn.Linear(cfg.hidden_dim, 3)

        if bool(getattr(cfg, "use_temporal_stop_state", True)):
            self.stop_module = TemporalStopTrendStateModule(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.stop_attention_heads,
                dropout=cfg.stop_attention_dropout,
                detach_context=cfg.stop_context_detach,
                temporal_window=getattr(cfg, "temporal_stop_window", 3),
                temporal_heads=getattr(cfg, "temporal_stop_heads", 4),
                temporal_dropout=getattr(cfg, "temporal_stop_dropout", cfg.dropout),
            )
            self.stop_head = None
            self.prestop_head = None
        elif bool(getattr(cfg, "use_stop_trend_state", True)):
            self.stop_module = StopTrendStateModule(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.stop_attention_heads,
                dropout=cfg.stop_attention_dropout,
                detach_context=cfg.stop_context_detach,
            )
            self.stop_head = None
            self.prestop_head = None
        else:
            self.stop_module = None
            self.stop_head = nn.Linear(cfg.hidden_dim, 1)
            self.prestop_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        progress: torch.Tensor = None,
        stop_history: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        shared_h = self.trunk(features.float())

        action_h = self.action_branch(shared_h)
        waypoint_h = self.waypoint_branch(shared_h)

        action_logits = self.action_head(action_h)
        waypoint_pred = self.waypoint_head(waypoint_h)

        if self.stop_module is not None:
            try:
                stop_out = self.stop_module(
                    shared_hidden=shared_h,
                    action_hidden=action_h,
                    waypoint_hidden=waypoint_h,
                    progress=progress,
                    stop_history=stop_history,
                )
            except TypeError:
                stop_out = self.stop_module(
                    shared_hidden=shared_h,
                    action_hidden=action_h,
                    waypoint_hidden=waypoint_h,
                    progress=progress,
                )

            stop_logit = stop_out.get("stop_logit", stop_out.get("single_stop_logit"))
            single_stop_logit = stop_out.get("single_stop_logit", stop_logit)
            base_stop_logit = stop_out["base_stop_logit"]
            prestop_logit = stop_out["prestop_logit"]
            stop_delta = stop_out["stop_delta"]
            stop_confidence = stop_out["stop_confidence"]
            stop_token = stop_out.get("stop_token", torch.zeros_like(action_h))
            temporal_stop_state = stop_out.get("temporal_stop_state", torch.zeros_like(action_h))
        else:
            stop_logit = self.stop_head(shared_h).squeeze(-1)
            single_stop_logit = stop_logit
            base_stop_logit = stop_logit
            prestop_logit = self.prestop_head(shared_h).squeeze(-1)
            stop_delta = torch.zeros_like(stop_logit)
            stop_confidence = torch.zeros_like(stop_logit)
            stop_token = torch.zeros_like(action_h)
            temporal_stop_state = torch.zeros_like(action_h)

        return {
            "action_logits": action_logits,
            "waypoint_pred": waypoint_pred,
            "stop_logit": stop_logit,
            "single_stop_logit": single_stop_logit,
            "base_stop_logit": base_stop_logit,
            "prestop_logit": prestop_logit,
            "temporal_stop_logit": stop_logit,
            "stop_delta": stop_delta,
            "stop_confidence": stop_confidence,
            "stop_token": stop_token,
            "temporal_stop_state": temporal_stop_state,
        }


def build_progress_from_batch(batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    vals = []
    for meta in batch.get("meta", []):
        cur = int(meta.get("current_idx", 0))
        steps_to_end = int(meta.get("steps_to_end", 0))
        denom = max(cur + steps_to_end, 1)
        vals.append(float(cur) / float(denom))

    if not vals:
        bsz = int(batch["action_label"].shape[0])
        vals = [0.0] * bsz

    return torch.tensor(vals, dtype=torch.float32, device=device).view(-1, 1)


def build_progress_from_step(step: int, max_step: int, device: torch.device) -> torch.Tensor:
    v = float(step) / float(max(max_step, 1))
    return torch.tensor([[v]], dtype=torch.float32, device=device)


class StopPrestopLabelBuilder:
    """
    Final STEER-VLN label builder.

    stop_label:
      strict commit-level stop label:
        action_label == 0 or steps_to_end <= 1

    prestop_label:
      broad near-terminal trend:
        steps_to_end <= horizon + prestop_window
    """

    def __init__(self, annotation_path: str, prestop_window: int = 3, horizon: int = 4):
        self.annotation_path = str(annotation_path)
        self.prestop_window = int(prestop_window)
        self.horizon = int(horizon)

        with open(annotation_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        self.actions_by_path = {}
        for item in items:
            image_path = str(item.get("image_path", ""))
            actions = item.get("action", [])
            if image_path and isinstance(actions, list):
                self.actions_by_path[image_path] = [int(x) for x in actions]

    def labels_from_batch(self, batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        action_label = batch["action_label"].detach().cpu().long()
        action_stop_label = (action_label == 0).float()

        metas = batch.get("meta", None)
        if metas is None:
            stop = action_stop_label.to(device)
            return stop, stop

        stop_values = []
        prestop_values = []

        for i, meta in enumerate(metas):
            steps_to_end = int(meta.get("steps_to_end", 999999))

            stop_label = bool(action_stop_label[i].item() > 0.5) or (steps_to_end <= 1)
            prestop_label = stop_label or (steps_to_end <= (self.horizon + self.prestop_window))

            stop_values.append(float(stop_label))
            prestop_values.append(float(prestop_label))

        return (
            torch.tensor(stop_values, dtype=torch.float32, device=device),
            torch.tensor(prestop_values, dtype=torch.float32, device=device),
        )


def focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    target = target.float()
    pos_weight_tensor = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight_tensor,
        reduction="none",
    )

    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)

    alpha_t = torch.where(
        target > 0.5,
        torch.full_like(target, alpha),
        torch.full_like(target, 1.0 - alpha),
    )

    return (alpha_t * torch.pow(1.0 - pt, gamma) * bce).mean()


def trend_head_loss(
    *,
    action_logits,
    action_label,
    waypoint_pred,
    waypoint_label,
    waypoint_mask,
    stop_logit,
    stop_label,
    prestop_logit,
    prestop_label,
    action_loss_weight=1.0,
    waypoint_loss_weight=0.1,
    stop_loss_weight=0.5,
    prestop_loss_weight=0.2,
    weight_d=1.0,
    weight_yaw=1.0,
    weight_z=0.0,
    stop_pos_weight=5.0,
    prestop_pos_weight=3.0,
    focal_gamma=2.0,
):
    action_loss = F.cross_entropy(action_logits, action_label.long())

    wp_diff = F.smooth_l1_loss(waypoint_pred, waypoint_label, reduction="none")
    wp_weights = torch.tensor(
        [weight_d, weight_yaw, weight_z],
        device=waypoint_pred.device,
        dtype=waypoint_pred.dtype,
    )
    wp_each = (wp_diff * wp_weights).sum(dim=-1)

    if waypoint_mask is not None:
        mask = waypoint_mask.float()
        waypoint_loss = (wp_each * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        waypoint_loss = wp_each.mean()

    stop_loss = focal_bce_with_logits(
        stop_logit,
        stop_label,
        alpha=0.75,
        gamma=focal_gamma,
        pos_weight=stop_pos_weight,
    )

    prestop_loss = focal_bce_with_logits(
        prestop_logit,
        prestop_label,
        alpha=0.70,
        gamma=focal_gamma,
        pos_weight=prestop_pos_weight,
    )

    total = (
        action_loss_weight * action_loss
        + waypoint_loss_weight * waypoint_loss
        + stop_loss_weight * stop_loss
        + prestop_loss_weight * prestop_loss
    )

    return {
        "loss": total,
        "action_loss": action_loss.detach(),
        "waypoint_loss": waypoint_loss.detach(),
        "stop_loss": stop_loss.detach(),
        "prestop_loss": prestop_loss.detach(),
    }


@torch.no_grad()
def binary_metrics_from_logits(logits, target, threshold=0.5, prefix="stop"):
    prob = torch.sigmoid(logits)
    pred = (prob >= threshold).long()
    target = target.long()

    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()

    total = max(tp + fp + fn + tn, 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        f"{prefix}_acc": float((tp + tn) / total),
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_f1": float(f1),
        f"{prefix}_pred_rate": float((tp + fp) / total),
        f"{prefix}_true_rate": float((tp + fn) / total),
        f"{prefix}_tp": float(tp),
        f"{prefix}_fp": float(fp),
        f"{prefix}_fn": float(fn),
        f"{prefix}_tn": float(tn),
        f"{prefix}_count": float(total),
    }


def aggregate_metric_dicts(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    out = {}
    for k in keys:
        vals = [float(x[k]) for x in items]
        if k.endswith(("_tp", "_fp", "_fn", "_tn", "_count")):
            out[k] = float(sum(vals))
        else:
            out[k] = float(sum(vals) / len(vals))
    return out


def save_trend_checkpoint(path, model, cfg, args, global_step, epoch, extra=None):
    payload = {
        "head": model.state_dict(),
        "cfg": cfg.__dict__,
        "args": vars(args) if hasattr(args, "__dict__") else {},
        "step": int(global_step),
        "epoch": int(epoch),
    }

    if extra:
        payload.update(extra)

    path = str(path)
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    Path(tmp_path).replace(path)


def load_trend_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu")
    cfg_dict = dict(ckpt["cfg"])

    cfg_dict.setdefault("use_stop_trend_state", True)
    cfg_dict.setdefault("use_temporal_stop_state", True)
    cfg_dict.setdefault("stop_attention_heads", 4)
    cfg_dict.setdefault("stop_attention_dropout", cfg_dict.get("dropout", 0.1))
    cfg_dict.setdefault("stop_context_detach", True)
    cfg_dict.setdefault("temporal_stop_window", 3)
    cfg_dict.setdefault("temporal_stop_heads", 4)
    cfg_dict.setdefault("temporal_stop_dropout", cfg_dict.get("dropout", 0.1))

    cfg = OpenFlyFeatureTrendHeadConfig(**cfg_dict)
    model = OpenFlyFeatureTrendHead(cfg)

    missing, unexpected = model.load_state_dict(ckpt["head"], strict=False)

    if missing:
        print(f"[LoadTrend][WARN] missing keys: {missing[:8]} ... total={len(missing)}", flush=True)
    if unexpected:
        print(f"[LoadTrend][WARN] unexpected keys: {unexpected[:8]} ... total={len(unexpected)}", flush=True)

    model.to(device)
    model.eval()
    return model, cfg, ckpt


def warmstart_from_dual_head(model, checkpoint_path: str) -> int:
    if not checkpoint_path:
        return 0

    p = Path(checkpoint_path)
    if not p.exists():
        print(f"[WarmStart][WARN] not found: {checkpoint_path}", flush=True)
        return 0

    ckpt = torch.load(str(p), map_location="cpu")
    source = ckpt.get("head", ckpt)

    own = model.state_dict()
    copied = 0

    for k, v in source.items():
        if k in own and tuple(own[k].shape) == tuple(v.shape):
            own[k].copy_(v)
            copied += 1

    model.load_state_dict(own, strict=False)
    print(f"[WarmStart] copied {copied} tensors from {checkpoint_path}", flush=True)
    return copied
