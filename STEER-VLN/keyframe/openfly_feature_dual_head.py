from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from keyframe.waypoint_probe_model import waypoint_smooth_l1_loss


@dataclass
class OpenFlyFeatureDualHeadConfig:
    input_dim: int
    hidden_dim: int = 512
    num_actions: int = 10
    dropout: float = 0.1


class OpenFlyFeatureDualHead(nn.Module):
    """
    External dual head on top of frozen OpenFly-Agent feature.

    Input:
        feature: [B, input_dim]

    Outputs:
        action_logits: [B, 10]
        waypoint_pred: [B, 3]
    """

    def __init__(self, cfg: OpenFlyFeatureDualHeadConfig):
        super().__init__()
        self.cfg = cfg

        self.shared = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),

            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.action_head = nn.Linear(cfg.hidden_dim, cfg.num_actions)

        self.waypoint_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 3),
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.zeros_(self.action_head.bias)

        last = self.waypoint_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.bias)

    def forward(self, feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.shared(feature)

        action_logits = self.action_head(hidden)

        wp_raw = self.waypoint_head(hidden)
        d = F.softplus(wp_raw[:, 0:1])
        yaw_z = torch.tanh(wp_raw[:, 1:3])
        waypoint_pred = torch.cat([d, yaw_z], dim=-1)

        return {
            "hidden": hidden,
            "action_logits": action_logits,
            "waypoint_pred": waypoint_pred,
        }


def dual_head_loss(
    action_logits: torch.Tensor,
    action_label: torch.Tensor,
    waypoint_pred: torch.Tensor,
    waypoint_label: torch.Tensor,
    waypoint_mask: torch.Tensor,
    action_loss_weight: float = 1.0,
    waypoint_loss_weight: float = 0.1,
    weight_d: float = 1.0,
    weight_yaw: float = 1.0,
    weight_z: float = 0.0,
) -> Dict[str, torch.Tensor]:
    action_loss = F.cross_entropy(action_logits, action_label)

    waypoint_loss = waypoint_smooth_l1_loss(
        pred=waypoint_pred,
        target=waypoint_label,
        mask=waypoint_mask,
        weight_d=weight_d,
        weight_yaw=weight_yaw,
        weight_z=weight_z,
    )

    total = action_loss_weight * action_loss + waypoint_loss_weight * waypoint_loss

    return {
        "loss": total,
        "action_loss": action_loss,
        "waypoint_loss": waypoint_loss,
    }


@torch.no_grad()
def action_metrics(action_logits: torch.Tensor, action_label: torch.Tensor) -> Dict[str, float]:
    pred = action_logits.argmax(dim=-1)
    acc = (pred == action_label).float().mean().item()
    return {"action_acc": float(acc)}
