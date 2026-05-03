"""
Attention-Augmented CNN for guided LST downscaling (4 km → 1 km).

Architecture:
  - LR branch: channel-attention residual blocks only. LR blocks are ~6×4
    pixels — convolutions already cover the whole patch, so spatial
    self-attention adds parameters without signal.
  - HR branch: same channel-attention-only design.
  - Fusion: cross-attention where each HR pixel (query) attends to the raw
    LR feature map (keys/values, ~24 tokens). Replaces bilinear upsample +
    concat + 1×1 conv; the model explicitly learns which LR context pixel
    is most relevant to each HR location.
  - Refinement: full spatial self-attention residual blocks at HR resolution
    (~330 tokens) where long-range terrain dependencies across the block are
    meaningful.
  - Output: residual on bicubic baseline (EDSR pattern).

Inputs at forward time:
  lr_lst   (B, 1, h, w)        LR LST at 4 km
  hr_cov   (B, C, H, W)        HR covariate stack at 1 km
  lr_mask  (B, 1, h, w)        1=valid, 0=cloud-filled (defaults to all-ones)
  lr_bicubic (B, 1, H, W)      pre-computed bicubic baseline from dataset
Output:
  hr_pred  (B, 1, H, W)        predicted HR LST at 1 km
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excite(w).view(b, c, 1, 1)
        return x * w


class SpatialSelfAttention(nn.Module):
    """Full multi-head self-attention over all spatial positions (H*W tokens)."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        seq = x.flatten(2).permute(2, 0, 1)        # (H*W, B, C)
        normed = self.norm(seq)
        attn_out, _ = self.attn(normed, normed, normed)
        attn_out = attn_out.permute(1, 2, 0).view(B, C, H, W)
        return x + attn_out


class ChannelResBlock(nn.Module):
    """Residual block with channel attention only — no spatial self-attention.

    Used in the LR and HR branches where spatial extent is too small for
    self-attention to add value over convolutions.
    GroupNorm keeps statistics stable on tiny LR patches (~6×4 pixels).
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.ca = ChannelAttention(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.ca(self.body(x)) + x)


class AttentionResBlock(nn.Module):
    """Residual block with channel attention then full spatial self-attention.

    Used only in the post-fusion refinement stage at HR resolution (~330 tokens)
    where long-range terrain dependencies across the block are meaningful.
    """

    def __init__(self, channels: int, num_heads: int = 8, groups: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.ca = ChannelAttention(channels)
        self.sa = SpatialSelfAttention(channels, num_heads)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        out = self.ca(out)
        out = self.sa(out)
        return self.relu(out + x)


class CrossAttentionFusion(nn.Module):
    """HR features (queries) attend to LR features (keys/values).

    Q sequence: HR feature map flattened → ~330 tokens.
    K/V sequence: LR feature map at native resolution → ~24 tokens.
    Each HR pixel learns which LR context token is most relevant,
    explicitly modelling the spatial LR→HR correspondence.

    attn_dropout randomly masks attention weights during training, preventing
    the model from memorising block-specific LR→HR routing patterns.
    """

    def __init__(self, channels: int, num_heads: int = 8, attn_dropout: float = 0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=False,
                                          dropout=attn_dropout)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, hr_feat: torch.Tensor, lr_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = hr_feat.shape
        q  = hr_feat.flatten(2).permute(2, 0, 1)   # (H*W, B, C)
        kv = lr_feat.flatten(2).permute(2, 0, 1)   # (h*w, B, C)
        kv_n = self.norm_kv(kv)
        out, _ = self.attn(self.norm_q(q), kv_n, kv_n)
        out = out.permute(1, 2, 0).view(B, C, H, W)
        return self.proj(hr_feat + out)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AttentionAugmentedCNN(nn.Module):
    """Guided super-resolution CNN with restructured attention.

    Attention is placed where it provides signal:
      - Channel attention in every residual block (cheap, effective everywhere).
      - Cross-attention at fusion (HR queries, LR keys/values).
      - Spatial self-attention only in post-fusion refinement blocks.

    Args:
        cov_channels:  number of HR-covariate channels (NDVI + DEM + LULC one-hot).
        base_channels: feature width.
        n_lr_blocks:   residual blocks at LR resolution.
        n_hr_blocks:   residual blocks at HR resolution (hr_body and refine).
        num_heads:     attention heads used in cross- and self-attention.
        dropout:       Dropout2d probability applied after fusion.
    """

    def __init__(
        self,
        cov_channels: int,
        base_channels: int = 64,
        n_lr_blocks: int = 4,
        n_hr_blocks: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        # --- LR branch: channel attention only ---
        self.lr_head = nn.Sequential(
            nn.Conv2d(2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lr_body = nn.Sequential(
            *[ChannelResBlock(base_channels) for _ in range(n_lr_blocks)]
        )

        # --- HR branch: channel attention only ---
        self.hr_head = nn.Sequential(
            nn.Conv2d(cov_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.hr_body = nn.Sequential(
            *[ChannelResBlock(base_channels) for _ in range(n_hr_blocks)]
        )

        # --- Fusion: HR queries attend to LR keys/values ---
        self.fuse = CrossAttentionFusion(base_channels, num_heads, attn_dropout=dropout)
        self.drop = nn.Dropout2d(p=dropout)

        # --- Post-fusion refinement: channel attention only ---
        self.refine = nn.Sequential(
            *[ChannelResBlock(base_channels) for _ in range(n_hr_blocks)]
        )
        self.tail = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, lr_lst: torch.Tensor, hr_cov: torch.Tensor,
                lr_mask: torch.Tensor | None = None,
                lr_bicubic: torch.Tensor | None = None) -> torch.Tensor:
        """
        lr_lst    : (B, 1, h, w)
        hr_cov    : (B, C, H, W)
        lr_mask   : (B, 1, h, w)  1=valid, 0=cloud-filled; defaults to all-ones
        lr_bicubic: (B, 1, H, W)  pre-computed bicubic baseline from dataset
        returns   : (B, 1, H, W)
        """
        H, W = hr_cov.shape[-2:]

        baseline = lr_bicubic if lr_bicubic is not None else \
                   F.interpolate(lr_lst, size=(H, W), mode="bicubic", align_corners=False)

        if lr_mask is None:
            lr_mask = torch.ones_like(lr_lst)

        lr_n    = lr_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1)
        lr_mean = (lr_lst * lr_mask).sum(dim=(-2, -1), keepdim=True) / lr_n
        lr_norm = (lr_lst - lr_mean) * lr_mask

        # LR features stay at LR resolution — cross-attention handles the mapping
        f_lr = self.lr_body(self.lr_head(torch.cat([lr_norm, lr_mask], dim=1)))
        f_hr = self.hr_body(self.hr_head(hr_cov))

        # Cross-attention fusion, then spatial self-attention refinement at HR
        fused = self.drop(self.fuse(f_hr, f_lr))
        delta = self.tail(self.refine(fused))

        return baseline + delta


# ---------------------------------------------------------------------------
# Convenience: assemble HR covariate tensor from a dataset batch
# ---------------------------------------------------------------------------

def make_hr_cov(batch: dict) -> torch.Tensor:
    """Standard HR-covariate stacking: [NDVI, DEM, LULC one-hot] → (B, C, H, W)."""
    return torch.cat([batch["ndvi"], batch["dem"], batch["lulc_onehot"]], dim=1)
