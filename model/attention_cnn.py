"""
Attention-Augmented CNN for temperature downscaling.

Architecture (from Section 4 of the project proposal):
  - Residual super-resolution backbone with dual attention
    (Channel Attention + Spatial Attention) in feature extraction blocks
  - Progressive upsampling via PixelShuffle: 1km -> 500m -> 250m (2x each stage)
  - Spatial attention reapplied after each upsampling stage

Input channels (concatenated at 1km):
  - Coarsened MODIS LST (bicubic-upsampled from 5km)  : 1 ch
  - DEM elevation                                      : 1 ch
  - DEM slope                                          : 1 ch
  - DEM aspect                                         : 1 ch
  - Sentinel-2 NDVI                                    : 1 ch
  - Sentinel-2 NDWI                                    : 1 ch
  - NLCD land cover (one-hot or embedded)               : N ch
                                                  Total: 6 + N
Output: 1-channel temperature field at 1km (5x spatial dimensions of input)
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention (SE block).

    Recalibrates inter-channel feature responses so the network can
    prioritize, e.g., DEM in alpine zones or NDVI over forests.
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


class SpatialAttention(nn.Module):
    """Spatial attention via channel-wise pooling + conv.

    Produces a learned spatial mask that focuses representational power
    on high-frequency terrain features (ridges, shadow boundaries) rather
    than homogeneous background.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        w = self.conv(torch.cat([avg, mx], dim=1))
        return x * w


class AttentionResBlock(nn.Module):
    """Residual block augmented with channel + spatial attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        out = self.ca(out)
        out = self.sa(out)
        return self.relu(out + x)


class UpsampleStage(nn.Module):
    """5x upsampling via PixelShuffle + spatial attention.

    Spatial attention is reapplied after upsampling to preserve
    terrain-driven temperature gradients at the finer resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, scale: int = 5):
        super().__init__()
        # PixelShuffle(r) needs r^2 × out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * scale ** 2, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.shuffle(self.conv(x)))
        out = self.sa(out)
        return out


class AttentionAugmentedCNN(nn.Module):
    """Attention-Augmented CNN for progressive temperature downscaling.

    Args:
        in_channels:  Number of input channels (LST + covariates).
        base_channels: Width of the feature extraction backbone.
        num_res_blocks: Number of attention-augmented residual blocks.
        scale_factor: Upsampling factor (5 = 5km -> 1km).
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 64,
        num_res_blocks: int = 8,
        scale_factor: int = 5,
    ):
        super().__init__()

        # --- Head: project input covariates into feature space ---
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # --- Body: stack of attention-augmented residual blocks ---
        self.body = nn.Sequential(
            *[AttentionResBlock(base_channels) for _ in range(num_res_blocks)]
        )
        # Global skip connection conv (long residual)
        self.body_tail = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        # --- Upsampling: 5x via PixelShuffle (5km -> 1km) ---
        self.upsample = UpsampleStage(base_channels, base_channels, scale=scale_factor)

        # --- Tail: map back to single temperature channel ---
        self.tail = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) tensor at 5km resolution.
               Channel order: [LST_coarse, DEM, slope, aspect, NDVI, NDWI, ...]

        Returns:
            (B, 1, H*5, W*5) downscaled temperature at 1km.
        """
        head = self.head(x)

        # Deep feature extraction with global residual
        body = self.body(head)
        body = self.body_tail(body) + head

        # Upsampling: 5km -> 1km
        up = self.upsample(body)

        return self.tail(up)
