"""Deterministic mock model used only by tests and smoke runs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class MockInterpolationModel(nn.Module):
    def __init__(
        self,
        *,
        output_scale: int = 4,
        endpoint_copy_box: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if output_scale < 1:
            raise ValueError("output_scale must be >= 1")
        if endpoint_copy_box is not None:
            if len(endpoint_copy_box) != 4:
                raise ValueError("endpoint_copy_box must contain x0,y0,x1,y1")
            values = tuple(float(value) for value in endpoint_copy_box)
            if not all(0.0 <= value <= 1.0 for value in values):
                raise ValueError("endpoint_copy_box coordinates must be normalized to [0,1]")
            if values[2] <= values[0] or values[3] <= values[1]:
                raise ValueError("endpoint_copy_box must have positive area")
            self.endpoint_copy_box = values
        else:
            self.endpoint_copy_box = None
        self.output_scale = int(output_scale)

    def forward(self, img0: torch.Tensor, img1: torch.Tensor):
        del img1
        batch, _, height, width = img0.shape
        low_height = max(1, height // self.output_scale)
        low_width = max(1, width // self.output_scale)
        flow = torch.zeros((batch, 2, low_height, low_width), device=img0.device, dtype=img0.dtype)
        mask0 = torch.full(
            (batch, 1, low_height, low_width), 0.5, device=img0.device, dtype=img0.dtype
        )
        mask1 = torch.zeros(
            (batch, 1, low_height, low_width), device=img0.device, dtype=img0.dtype
        )
        if self.endpoint_copy_box is not None:
            x0, y0, x1, y1 = self.endpoint_copy_box
            ix0 = min(low_width - 1, max(0, int(x0 * low_width)))
            iy0 = min(low_height - 1, max(0, int(y0 * low_height)))
            ix1 = min(low_width, max(ix0 + 1, int(x1 * low_width + 0.999)))
            iy1 = min(low_height, max(iy0 + 1, int(y1 * low_height + 0.999)))
            mask1[:, :, iy0:iy1, ix0:ix1] = 1.0
        return {
            "flow_t0": flow,
            "flow_t1": flow.clone(),
            "mask0": mask0,
            "mask1": mask1,
        }


def create_model(
    *,
    checkpoint: str | None,
    device: torch.device,
    output_scale: int = 4,
    endpoint_copy_box: Sequence[float] | None = None,
) -> MockInterpolationModel:
    """Factory matching the production adapter contract."""

    if checkpoint not in (None, ""):
        raise ValueError("the mock model does not load checkpoints")
    return MockInterpolationModel(
        output_scale=output_scale,
        endpoint_copy_box=endpoint_copy_box,
    ).to(device)
