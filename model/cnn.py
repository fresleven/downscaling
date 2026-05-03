"""
Plain guided-SR CNN baseline (no attention).

Same dual-branch / fusion / residual-on-bicubic structure as
`attention_cnn.AttentionAugmentedCNN`, but with vanilla residual blocks —
serves as the "what does attention buy us?" control.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Plain residual block: conv-GN-ReLU-conv-GN + identity.

    GroupNorm (groups=8) instead of BatchNorm so statistics are stable
    even on the tiny LR patches (~7×6 pixels) seen at LR resolution.
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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + x)


class GuidedCNN(nn.Module):
    """Guided-SR CNN: LR LST + HR covariates → HR LST. Predicts residual on bicubic.

    Args:
        cov_channels: number of HR-covariate channels (NDVI + DEM + LULC one-hot).
        base_channels: feature width.
        n_lr_blocks: residual blocks at LR resolution.
        n_hr_blocks: residual blocks at HR resolution after fusion.
    """

    def __init__(
        self,
        cov_channels: int,
        base_channels: int = 64,
        n_lr_blocks: int = 4,
        n_hr_blocks: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        # 2 input channels: LR LST (mean-subtracted) + LR cloud mask (1=valid, 0=cloud-filled)
        self.lr_head = nn.Sequential(
            nn.Conv2d(2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lr_body = nn.Sequential(*[ResBlock(base_channels) for _ in range(n_lr_blocks)])

        self.hr_head = nn.Sequential(
            nn.Conv2d(cov_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.hr_body = nn.Sequential(*[ResBlock(base_channels) for _ in range(n_hr_blocks)])

        self.fuse = nn.Conv2d(2 * base_channels, base_channels, 1)
        self.drop = nn.Dropout2d(p=dropout)
        self.refine = nn.Sequential(*[ResBlock(base_channels) for _ in range(n_hr_blocks)])
        self.tail = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, lr_lst: torch.Tensor, hr_cov: torch.Tensor,
                lr_mask: torch.Tensor | None = None,
                lr_bicubic: torch.Tensor | None = None) -> torch.Tensor:
        H, W = hr_cov.shape[-2:]

        # Prefer the pre-computed correct-scale baseline from the dataset.
        # Fallback: upsample the (possibly padded) LR — only correct when padding=0.
        baseline = lr_bicubic if lr_bicubic is not None else \
                   F.interpolate(lr_lst, size=(H, W), mode="bicubic", align_corners=False)

        if lr_mask is None:
            lr_mask = torch.ones_like(lr_lst)

        # Mean over valid (non-padded, non-cloudy) LR pixels only.
        # Using .mean() over all positions is wrong when padding zeros pull the
        # mean far from the true block mean, corrupting the normalized features.
        lr_n    = lr_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1)
        lr_mean = (lr_lst * lr_mask).sum(dim=(-2, -1), keepdim=True) / lr_n
        lr_norm = (lr_lst - lr_mean) * lr_mask   # zero out padded/cloudy positions

        f_lr = self.lr_body(self.lr_head(torch.cat([lr_norm, lr_mask], dim=1)))
        f_lr_up = F.interpolate(f_lr, size=(H, W), mode="bilinear", align_corners=False)

        f_hr = self.hr_body(self.hr_head(hr_cov))
        fused = self.drop(self.fuse(torch.cat([f_lr_up, f_hr], dim=1)))
        delta = self.tail(self.refine(fused))
        return baseline + delta
