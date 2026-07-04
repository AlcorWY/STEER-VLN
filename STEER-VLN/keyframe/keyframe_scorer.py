from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KeyframeScorerConfig:
    image_size: int = 224
    vocab_size: int = 32000
    text_dim: int = 256
    hidden_dim: int = 256
    motion_dim: int = 4
    max_history: int = 8
    num_heads: int = 4
    dropout: float = 0.1
    use_temporal_self_attention: bool = True
    use_motion_residual: bool = True


class SmallFrameEncoder(nn.Module):
    """
    Lightweight CNN image encoder.
    输入: [B, 3, H, W]
    输出: [B, hidden_dim]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.flatten(1)


class TextMeanEncoder(nn.Module):
    """
    Lightweight instruction encoder.
    用 tokenizer 的 input_ids 做 embedding mean pooling。
    """

    def __init__(self, vocab_size: int, text_dim: int, hidden_dim: int, padding_idx: int = 0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, text_dim, padding_idx=padding_idx)
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)

        mask = attention_mask.unsqueeze(-1).to(emb.dtype)
        emb = emb * mask

        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = emb.sum(dim=1) / denom

        return self.proj(pooled)


class AttentionKeyframeScorer(nn.Module):
    """
    三段式 learned keyframe scorer。

    A. FrameEncoder 编码历史候选帧和当前帧
    B. TextEncoder + current frame 构造 query；motion residual 注入历史 token
    C. temporal self-attention + query/history dot scoring
    """

    def __init__(self, cfg: KeyframeScorerConfig, padding_idx: int = 0):
        super().__init__()

        self.cfg = cfg

        self.frame_encoder = SmallFrameEncoder(cfg.hidden_dim)
        self.text_encoder = TextMeanEncoder(
            vocab_size=cfg.vocab_size,
            text_dim=cfg.text_dim,
            hidden_dim=cfg.hidden_dim,
            padding_idx=padding_idx,
        )

        self.query_fuser = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        self.time_embedding = nn.Embedding(cfg.max_history, cfg.hidden_dim)

        if cfg.use_motion_residual:
            self.motion_encoder = nn.Sequential(
                nn.Linear(cfg.motion_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
        else:
            self.motion_encoder = None

        if cfg.use_temporal_self_attention:
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.hidden_dim * 4,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=1)
        else:
            self.temporal_encoder = None

        self.score_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(
        self,
        history_images: torch.Tensor,
        current_image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        motion_feats: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        history_images: [B, K, 3, H, W]
        current_image: [B, 3, H, W]
        input_ids: [B, L]
        attention_mask: [B, L]
        motion_feats: [B, K, 4]
        history_mask: [B, K], True for valid candidates

        return:
            logits: [B, K]
        """

        bsz, k, c, h, w = history_images.shape

        hist_flat = history_images.view(bsz * k, c, h, w)
        hist_feat = self.frame_encoder(hist_flat).view(bsz, k, -1)

        cur_feat = self.frame_encoder(current_image)
        txt_feat = self.text_encoder(input_ids, attention_mask)

        query = self.query_fuser(torch.cat([cur_feat, txt_feat], dim=-1))

        pos_ids = torch.arange(k, device=history_images.device).unsqueeze(0).expand(bsz, k)
        hist_feat = hist_feat + self.time_embedding(pos_ids)

        if self.motion_encoder is not None:
            hist_feat = hist_feat + self.motion_encoder(motion_feats)

        if self.temporal_encoder is not None:
            key_padding_mask = ~history_mask.bool()
            hist_feat = self.temporal_encoder(hist_feat, src_key_padding_mask=key_padding_mask)

        query_expand = query.unsqueeze(1).expand(-1, k, -1)
        prod = query_expand * hist_feat

        logits = self.score_proj(torch.cat([query_expand, hist_feat, prod], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~history_mask.bool(), -1e4)

        return logits


def keyframe_soft_ce_loss(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
    history_mask: torch.Tensor,
) -> torch.Tensor:
    """
    soft_labels: [B, K], sum over valid candidates = 1
    """
    logits = logits.masked_fill(~history_mask.bool(), -1e4)
    log_probs = F.log_softmax(logits, dim=-1)

    loss = -(soft_labels * log_probs).sum(dim=-1)
    return loss.mean()


@torch.no_grad()
def keyframe_top1_accuracy(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
    history_mask: torch.Tensor,
) -> float:
    logits = logits.masked_fill(~history_mask.bool(), -1e4)
    pred = logits.argmax(dim=-1)
    target = soft_labels.argmax(dim=-1)

    valid = history_mask.any(dim=-1)
    if valid.sum().item() == 0:
        return 0.0

    acc = (pred[valid] == target[valid]).float().mean().item()
    return acc
