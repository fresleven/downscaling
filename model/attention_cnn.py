"""
Attention-Augmented CNN for guided LST downscaling (4 km → 1 km).

Architecture:
  - LR branch: feature extraction on the 4 km LST input through residual
    blocks with channel attention + full spatial self-attention.
  - HR branch: feature extraction on the 1 km guidance covariates
    (NDVI + DEM + LULC one-hot) — same residual+attention design.
  - Fusion: bilinear-upsample LR features to the HR grid, concatenate with
    HR features, refine with a stack of attention-augmented residual blocks
    at HR resolution.
  - Output: predict the *residual* on top of a bicubic-upsampled LR baseline.
    The network only has to learn the high-frequency correction; the low-
    frequency component is preserved by construction. Standard EDSR pattern.

Spatial attention is full multi-head self-attention over all H×W positions
(no masking). With block-as-sample, HR blocks are ~22×14 ≈ 300 tokens and LR
scenes are ~28×22 ≈ 600 tokens — both small enough for dense attention.

Inputs at forward time:
  lr_lst   (B, 1, h, w)        LR LST at 4 km                  (e.g. 28x22)
  hr_cov   (B, C, H, W)        HR covariate stack at 1 km      (e.g. 112x87)
                               channel order is up to the caller; suggested:
                               [NDVI(1), DEM(1), LULC_onehot(N)]
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
    """Squeeze-and-Excitation channel attention.

    Recalibrates inter-channel responses so the network can prioritize, e.g.,
    DEM in alpine zones or NDVI over forest, conditional on the input.
    """

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
    """Full multi-head self-attention over all spatial positions (H*W tokens).

    Each pixel attends to every other pixel in the feature map. Pre-norm
    with an additive residual — standard transformer block style.
    """

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


class AttentionResBlock(nn.Module):
    """Residual block with channel attention then full spatial self-attention."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.ca = ChannelAttention(channels)
        self.sa = SpatialSelfAttention(channels, num_heads)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        out = self.ca(out)
        out = self.sa(out)
        return self.relu(out + x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AttentionAugmentedCNN(nn.Module):
    """Guided super-resolution CNN with dual attention.

    Args:
        cov_channels:    number of HR-covariate channels (NDVI + DEM + LULC one-hot).
        base_channels:   feature width.
        n_lr_blocks:     residual blocks at LR resolution.
        n_hr_blocks:     residual blocks at HR resolution after fusion.
    """

    def __init__(
        self,
        cov_channels: int,
        base_channels: int = 64,
        n_lr_blocks: int = 4,
        n_hr_blocks: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()

        def _block():
            return AttentionResBlock(base_channels, num_heads)

        # --- LR branch (operates on the 4 km LST) ---
        self.lr_head = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lr_body = nn.Sequential(*[_block() for _ in range(n_lr_blocks)])

        # --- HR branch (operates on covariates at 1 km) ---
        self.hr_head = nn.Sequential(
            nn.Conv2d(cov_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.hr_body = nn.Sequential(*[_block() for _ in range(n_lr_blocks)])

        # --- Fusion + HR refinement ---
        self.fuse = nn.Conv2d(2 * base_channels, base_channels, 1)
        self.refine = nn.Sequential(*[_block() for _ in range(n_hr_blocks)])
        self.tail = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, lr_lst: torch.Tensor, hr_cov: torch.Tensor) -> torch.Tensor:
        """
        lr_lst : (B, 1, h, w)
        hr_cov : (B, C, H, W)
        returns: (B, 1, H, W)
        """
        H, W = hr_cov.shape[-2:]

        # Bicubic baseline — guarantees the network preserves low-freq content.
        baseline = F.interpolate(lr_lst, size=(H, W), mode="bicubic", align_corners=False)

        # LR feature path
        f_lr = self.lr_body(self.lr_head(lr_lst))
        f_lr_up = F.interpolate(f_lr, size=(H, W), mode="bilinear", align_corners=False)

        # HR feature path
        f_hr = self.hr_body(self.hr_head(hr_cov))

        # Fuse and refine at HR
        fused = self.fuse(torch.cat([f_lr_up, f_hr], dim=1))
        refined = self.refine(fused)
        delta = self.tail(refined)

        return baseline + delta


# ---------------------------------------------------------------------------
# Convenience: assemble HR covariate tensor from a dataset batch
# ---------------------------------------------------------------------------

def make_hr_cov(batch: dict) -> torch.Tensor:
    """Standard HR-covariate stacking: [NDVI, DEM, LULC one-hot] → (B, C, H, W)."""
    return torch.cat([batch["ndvi"], batch["dem"], batch["lulc_onehot"]], dim=1)
