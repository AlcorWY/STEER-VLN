from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WaypointProbeConfig:
    image_size: int = 224
    vocab_size: int = 20000
    text_dim: int = 128
    hidden_dim: int = 256
    motion_dim: int = 8
    num_frames: int = 3
    dropout: float = 0.1


class SmallImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.cnn(x))


class TextMeanEncoder(nn.Module):
    def __init__(self, vocab_size: int, text_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, text_dim, padding_idx=padding_idx)
        self.proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)

        mask = attention_mask.float().unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)

        pooled = summed / denom
        return self.proj(pooled)


class WaypointProbe(nn.Module):
    """
    External lightweight waypoint probe.

    It does not require OpenFly-Agent 7B.
    It consumes:
        images: [B, 3, 3, H, W], order [keyframe, previous, current]
        instruction tokens
        motion_feats: [B, 8]

    It predicts:
        waypoint_norm: [B, 3] = [Δd/d_scale, Δyaw/pi, Δz/z_scale]
    """

    def __init__(self, cfg: WaypointProbeConfig, padding_idx: int = 0):
        super().__init__()
        self.cfg = cfg

        self.image_encoder = SmallImageEncoder(out_dim=cfg.hidden_dim)
        self.text_encoder = TextMeanEncoder(
            vocab_size=cfg.vocab_size,
            text_dim=cfg.text_dim,
            padding_idx=padding_idx,
        )

        self.frame_type_embed = nn.Parameter(torch.zeros(cfg.num_frames, cfg.hidden_dim))

        fusion_dim = cfg.hidden_dim * cfg.num_frames + cfg.text_dim + cfg.motion_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, cfg.hidden_dim * 2),
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),

            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.waypoint_head = nn.Linear(cfg.hidden_dim, 3)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.frame_type_embed, std=0.02)
        nn.init.zeros_(self.waypoint_head.bias)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        motion_feats: torch.Tensor,
    ) -> torch.Tensor:
        b, n, c, h, w = images.shape

        flat = images.view(b * n, c, h, w)
        img_feat = self.image_encoder(flat)
        img_feat = img_feat.view(b, n, -1)

        img_feat = img_feat + self.frame_type_embed.unsqueeze(0)

        img_flat = img_feat.reshape(b, -1)

        text_feat = self.text_encoder(input_ids, attention_mask)

        fused = torch.cat([img_flat, text_feat, motion_feats], dim=-1)
        hidden = self.fusion(fused)

        pred = self.waypoint_head(hidden)

        # Keep ranges stable:
        # Δd_norm roughly non-negative; Δyaw_norm / Δz_norm bounded.
        d = F.softplus(pred[:, 0:1])
        yaw_z = torch.tanh(pred[:, 1:3])

        return torch.cat([d, yaw_z], dim=-1)


def waypoint_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight_d: float = 1.0,
    weight_yaw: float = 1.0,
    weight_z: float = 0.5,
) -> torch.Tensor:
    per_dim = F.smooth_l1_loss(pred, target, reduction="none")

    weights = torch.tensor(
        [weight_d, weight_yaw, weight_z],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 3)

    loss = (per_dim * weights).sum(dim=-1)

    mask = mask.float()
    loss = loss * mask

    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


@torch.no_grad()
def waypoint_metrics(
    pred: torch.Tensor,
    target_norm: torch.Tensor,
    target_raw: torch.Tensor,
    mask: torch.Tensor,
    d_scale: float = 9.0,
    z_scale: float = 5.0,
    yaw_sign_threshold: float = 0.10,
):
    mask = mask.bool()

    if mask.sum().item() == 0:
        return {
            "mae_d": 0.0,
            "mae_yaw": 0.0,
            "mae_z": 0.0,
            "norm_l1": 0.0,
            "yaw_sign_acc": 0.0,
            "yaw_sign_count": 0,
        }

    p = pred[mask]
    t_norm = target_norm[mask]
    t_raw = target_raw[mask]

    pred_raw = torch.zeros_like(t_raw)
    pred_raw[:, 0] = p[:, 0] * d_scale
    pred_raw[:, 1] = p[:, 1] * torch.pi
    pred_raw[:, 2] = p[:, 2] * z_scale

    mae = torch.abs(pred_raw - t_raw).mean(dim=0)

    norm_l1 = torch.abs(p - t_norm).mean().item()

    valid_yaw = torch.abs(t_raw[:, 1]) > yaw_sign_threshold
    if valid_yaw.sum().item() > 0:
        sign_acc = (
            torch.sign(pred_raw[valid_yaw, 1])
            == torch.sign(t_raw[valid_yaw, 1])
        ).float().mean().item()
        sign_count = int(valid_yaw.sum().item())
    else:
        sign_acc = 0.0
        sign_count = 0

    return {
        "mae_d": float(mae[0].item()),
        "mae_yaw": float(mae[1].item()),
        "mae_z": float(mae[2].item()),
        "norm_l1": float(norm_l1),
        "yaw_sign_acc": float(sign_acc),
        "yaw_sign_count": sign_count,
    }
