"""Lightweight UNet for midpoint optical flow and blending mask estimation.

Architecture overview
---------------------
Input  : cat(img0, img1)  →  [B, 6, H, W]
Encoder: 3 downsampling stages (×2 each)
         enc1 → [B,   C, H,   W  ]
         enc2 → [B,  2C, H/2, W/2]
         enc3 → [B,  4C, H/4, W/4]
Bottleneck:      [B,  8C, H/8, W/8]
Decoder: 1 stage  → [B, 4C, H/4, W/4]  (quarter-resolution output)
Head   : 1×1 conv → 6 channels
         ch 0-1  →  flow_t0   [B, 2, h, w]
         ch 2-3  →  flow_t1   [B, 2, h, w]
         ch 4    →  mask0     [B, 1, h, w]  (sigmoid-normalized)
         ch 5    →  mask1     [B, 1, h, w]  (sigmoid-normalized)

Config entry
------------
    "model": {
        "factory": "unet",
        "checkpoint": null,
        "input_height": 540,
        "input_width":  960,
        "batch_size":   4,
        "factory_kwargs": { "base_channels": 32 }
    }

The output spatial size is input_height/4 × input_width/4 (e.g. 135×240 for
a 540×960 input).  ModelAdapter handles all resizing before calling forward().
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Private building blocks
# ---------------------------------------------------------------------------

class _ConvBnRelu(nn.Sequential):
    """3×3 Conv → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ConvBlock(nn.Sequential):
    """Two stacked _ConvBnRelu layers — standard UNet double-conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            _ConvBnRelu(in_ch, out_ch),
            _ConvBnRelu(out_ch, out_ch),
        )


class _DownBlock(nn.Module):
    """2×2 MaxPool (ceil_mode) followed by a double-conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _UpBlock(nn.Module):
    """Transposed conv upsample + skip connection + double-conv.

    A bilinear correction handles any 1-pixel size mismatch that arises
    from ceil_mode pooling on odd-dimensional inputs.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------

class UNetModel(nn.Module):
    """UNet-based VFI model outputting quarter-resolution flows and masks.

    Parameters
    ----------
    base_channels:
        Width multiplier for all feature maps.  The four encoder stages
        use C, 2C, 4C, 8C channels respectively.  Default 32 gives a
        lightweight network suitable for experimentation; increase to 64
        or 96 for production capacity.
    """

    def __init__(self, *, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        # Encoder — receives concatenated frames [B, 6, H, W]
        self.enc1 = _ConvBlock(6, c)          # [B,  c, H,   W  ]
        self.enc2 = _DownBlock(c, c * 2)      # [B, 2c, H/2, W/2]
        self.enc3 = _DownBlock(c * 2, c * 4)  # [B, 4c, H/4, W/4]

        # Bottleneck
        self.bottleneck = _DownBlock(c * 4, c * 8)  # [B, 8c, H/8, W/8]

        # Decoder — one stage back to H/4 (quarter resolution)
        self.dec = _UpBlock(c * 8, c * 4, c * 4)   # [B, 4c, H/4, W/4]

        # Output head: flow_t0(2) + flow_t1(2) + mask0(1) + mask1(1) = 6
        self.head = nn.Conv2d(c * 4, 6, kernel_size=1)

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Estimate midpoint optical flow and blending masks.

        Called by ``ModelAdapter`` after it has already resized inputs to
        ``(input_height, input_width)`` and moved them to the target device.

        Parameters
        ----------
        img0, img1:
            ``[B, 3, H, W]`` float32, RGB normalized to ``[0, 1]``.

        Returns
        -------
        dict with keys:

        ``flow_t0``  ``[B, 2, H/4, W/4]``  backward flow from t=0 to midpoint
        ``flow_t1``  ``[B, 2, H/4, W/4]``  backward flow from t=1 to midpoint
        ``mask0``    ``[B, 1, H/4, W/4]``  blending weight in ``[0, 1]``
        ``mask1``    ``[B, 1, H/4, W/4]``  blending weight in ``[0, 1]``
        """
        x = torch.cat([img0, img1], dim=1)  # [B, 6, H, W]

        s1 = self.enc1(x)           # [B,  c, H,   W  ]
        s2 = self.enc2(s1)          # [B, 2c, H/2, W/2]
        s3 = self.enc3(s2)          # [B, 4c, H/4, W/4]
        b  = self.bottleneck(s3)    # [B, 8c, H/8, W/8]
        d  = self.dec(b, s3)        # [B, 4c, H/4, W/4]

        out = self.head(d)          # [B, 6, H/4, W/4]

        return {
            "flow_t0": out[:, 0:2],
            "flow_t1": out[:, 2:4],
            "mask0":   torch.sigmoid(out[:, 4:5]),
            "mask1":   torch.sigmoid(out[:, 5:6]),
        }

    @torch.inference_mode()
    def infer(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run a single forward pass with gradient tracking disabled.

        Use this when calling the model directly (e.g. in a custom
        evaluation script).  ``ModelAdapter`` calls ``forward()``
        internally and wraps it in ``torch.inference_mode()`` itself,
        so you do not need to call ``infer()`` when going through the
        adapter.

        Parameters
        ----------
        img0, img1:
            ``[B, 3, H, W]`` float32 tensors already on the correct
            device, RGB normalized to ``[0, 1]``.  No resizing is
            applied here — pass the network's fixed ``input_height ×
            input_width`` directly.
        """
        return self(img0, img1)


# ---------------------------------------------------------------------------
# Factory (called by ModelAdapter via load_factory)
# ---------------------------------------------------------------------------

def create_model(
    *,
    checkpoint: str | None,
    device: torch.device,
    base_channels: int = 32,
) -> UNetModel:
    """Construct a UNetModel, optionally load weights, and return it eval-ready.

    Parameters
    ----------
    checkpoint:
        Path to a ``state_dict`` saved with ``torch.save(model.state_dict(),
        path)``, or ``None`` for random initialization.
    device:
        Target device, resolved by ``ModelAdapter`` from the runtime config.
    base_channels:
        Forwarded from ``model.factory_kwargs.base_channels`` in the config.
        Must match the value used when the checkpoint was trained.
    """
    model = UNetModel(base_channels=base_channels)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model.eval().to(device)
