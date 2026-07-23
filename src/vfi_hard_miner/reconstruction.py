"""Deterministic reconstruction for midpoint frame interpolation.

The production model returns backward flows and two already-sigmoided masks.
Reconstruction runs in float32 on a caller-selected device: the CPU path is
the calibrated numerical reference, and accelerator devices (CUDA/NPU) may be
used for throughput when their ``grid_sample`` support has been probed.  With
``device=None`` or ``device="cpu"`` every operation is identical to the
original CPU reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


Mask0Role = Literal["warp0_weight", "warp1_weight"]
PaddingMode = Literal["zeros", "border", "reflection"]


@dataclass(frozen=True)
class ReconstructionResult:
    """Original-resolution float32 tensors used for diagnosis.

    Tensors live on the reconstruction device until the caller transfers them
    (see ``pack_reconstruction_to_cpu``).
    """

    flow_t0: torch.Tensor
    flow_t1: torch.Tensor
    mask0: torch.Tensor
    mask1: torch.Tensor
    warp0: torch.Tensor
    warp1: torch.Tensor
    warp_blend: torch.Tensor
    prediction: torch.Tensor


DeviceLike = torch.device | str | None


def _resolve_device(device: DeviceLike) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def _to_device_float32(tensor: torch.Tensor, device: torch.device, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    return tensor.detach().to(device=device, dtype=torch.float32)


def _as_cpu_float32(tensor: torch.Tensor, name: str) -> torch.Tensor:
    return _to_device_float32(tensor, torch.device("cpu"), name)


def _validate_nchw(
    tensor: torch.Tensor,
    name: str,
    *,
    channels: int,
    batch: int | None = None,
    spatial: tuple[int, int] | None = None,
) -> None:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B,{channels},H,W], got {tuple(tensor.shape)}")
    if tensor.shape[1] != channels:
        raise ValueError(
            f"{name} must have {channels} channels, got shape {tuple(tensor.shape)}"
        )
    if tensor.shape[0] <= 0 or tensor.shape[2] <= 0 or tensor.shape[3] <= 0:
        raise ValueError(f"{name} dimensions must be positive, got {tuple(tensor.shape)}")
    if batch is not None and tensor.shape[0] != batch:
        raise ValueError(f"{name} batch must be {batch}, got {tensor.shape[0]}")
    if spatial is not None and tuple(tensor.shape[-2:]) != tuple(spatial):
        raise ValueError(
            f"{name} spatial shape must be {tuple(spatial)}, got {tuple(tensor.shape[-2:])}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or infinity")


def _validate_unit_interval(tensor: torch.Tensor, name: str, tolerance: float = 1e-6) -> None:
    minimum = float(tensor.amin())
    maximum = float(tensor.amax())
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"{name} must be in [0,1], observed range [{minimum:.8g},{maximum:.8g}]"
        )


def _validate_hw(size: tuple[int, int], name: str) -> tuple[int, int]:
    if len(size) != 2:
        raise ValueError(f"{name} must contain (height, width), got {size!r}")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} dimensions must be positive, got {(height, width)}")
    return height, width


def resize_backward_flow(
    flow: torch.Tensor,
    output_size: tuple[int, int],
    network_size: tuple[int, int],
    *,
    align_corners: bool = False,
    device: DeviceLike = None,
) -> torch.Tensor:
    """Resize backward flow and convert network-input pixels to output pixels.

    Flow values are defined in the coordinate system of the fixed network
    input, regardless of the lower spatial resolution at which the model emits
    them.  Consequently x/y are scaled by ``W_out/W_network`` and
    ``H_out/H_network`` after interpolation.
    """

    target = _resolve_device(device)
    flow_device = _to_device_float32(flow, target, "flow")
    _validate_nchw(flow_device, "flow", channels=2)
    out_h, out_w = _validate_hw(output_size, "output_size")
    net_h, net_w = _validate_hw(network_size, "network_size")

    resized = F.interpolate(
        flow_device,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=align_corners,
    ).clone()
    resized[:, 0].mul_(out_w / net_w)
    resized[:, 1].mul_(out_h / net_h)
    if not bool(torch.isfinite(resized).all()):
        raise ValueError("resized flow contains NaN or infinity")
    return resized


def resize_mask(
    mask: torch.Tensor,
    output_size: tuple[int, int],
    *,
    align_corners: bool = False,
    device: DeviceLike = None,
) -> torch.Tensor:
    """Resize an already-sigmoided mask without changing its values otherwise."""

    target = _resolve_device(device)
    mask_device = _to_device_float32(mask, target, "mask")
    _validate_nchw(mask_device, "mask", channels=1)
    _validate_unit_interval(mask_device, "mask")
    out_h, out_w = _validate_hw(output_size, "output_size")
    resized = F.interpolate(
        mask_device,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=align_corners,
    )
    _validate_unit_interval(resized, "resized mask", tolerance=2e-6)
    return resized


def _pixel_to_normalized_grid(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    height: int,
    width: int,
    align_corners: bool,
) -> torch.Tensor:
    if align_corners:
        x_norm = torch.zeros_like(x) if width == 1 else (2.0 * x / (width - 1)) - 1.0
        y_norm = torch.zeros_like(y) if height == 1 else (2.0 * y / (height - 1)) - 1.0
    else:
        x_norm = ((2.0 * x + 1.0) / width) - 1.0
        y_norm = ((2.0 * y + 1.0) / height) - 1.0
    return torch.stack((x_norm, y_norm), dim=-1)


_MESHGRID_CACHE_MAX = 4
_meshgrid_cache: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _base_grid(
    height: int, width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached identity sampling grid; rebuilding it per warp wastes 4K memory traffic."""

    key = (height, width, str(device))
    cached = _meshgrid_cache.get(key)
    if cached is not None:
        return cached
    if len(_meshgrid_cache) >= _MESHGRID_CACHE_MAX:
        _meshgrid_cache.clear()
    y_base, x_base = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    _meshgrid_cache[key] = (y_base, x_base)
    return y_base, x_base


def backward_warp(
    image: torch.Tensor,
    backward_flow: torch.Tensor,
    *,
    align_corners: bool = False,
    padding_mode: PaddingMode = "border",
    device: DeviceLike = None,
) -> torch.Tensor:
    """Warp ``image`` using target-to-source pixel displacement.

    At output pixel ``(x, y)``, the source is sampled at
    ``(x + flow_x, y + flow_y)``.
    """

    if padding_mode not in {"zeros", "border", "reflection"}:
        raise ValueError(
            "padding_mode must be one of 'zeros', 'border', or 'reflection', "
            f"got {padding_mode!r}"
        )
    target = _resolve_device(device)
    image_device = _to_device_float32(image, target, "image")
    flow_device = _to_device_float32(backward_flow, target, "backward_flow")
    _validate_nchw(image_device, "image", channels=3)
    _validate_nchw(
        flow_device,
        "backward_flow",
        channels=2,
        batch=image_device.shape[0],
        spatial=tuple(image_device.shape[-2:]),
    )

    batch, _, height, width = image_device.shape
    y_base, x_base = _base_grid(height, width, image_device.device)
    x = x_base.unsqueeze(0).expand(batch, -1, -1) + flow_device[:, 0]
    y = y_base.unsqueeze(0).expand(batch, -1, -1) + flow_device[:, 1]
    grid = _pixel_to_normalized_grid(
        x,
        y,
        height=height,
        width=width,
        align_corners=align_corners,
    )
    warped = F.grid_sample(
        image_device,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    if not bool(torch.isfinite(warped).all()):
        raise ValueError("warped image contains NaN or infinity")
    return warped


def reconstruct_midpoint(
    img0: torch.Tensor,
    img1: torch.Tensor,
    flow_t0: torch.Tensor,
    flow_t1: torch.Tensor,
    mask0: torch.Tensor,
    mask1: torch.Tensor,
    *,
    network_size: tuple[int, int],
    mask0_role: Mask0Role,
    align_corners: bool = False,
    padding_mode: PaddingMode = "border",
    device: DeviceLike = None,
) -> ReconstructionResult:
    """Reconstruct the midpoint prediction under the fixed model contract.

    ``device=None`` or ``"cpu"`` runs the calibrated CPU reference path; any
    other device runs the same operations there (result tensors stay on that
    device until the caller transfers them).
    """

    if mask0_role not in {"warp0_weight", "warp1_weight"}:
        raise ValueError(
            "mask0_role must be 'warp0_weight' or 'warp1_weight'; "
            f"got {mask0_role!r}"
        )

    target = _resolve_device(device)
    image0 = _to_device_float32(img0, target, "img0")
    image1 = _to_device_float32(img1, target, "img1")
    _validate_nchw(image0, "img0", channels=3)
    _validate_nchw(
        image1,
        "img1",
        channels=3,
        batch=image0.shape[0],
        spatial=tuple(image0.shape[-2:]),
    )
    _validate_unit_interval(image0, "img0")
    _validate_unit_interval(image1, "img1")

    raw_flow0 = _to_device_float32(flow_t0, target, "flow_t0")
    raw_flow1 = _to_device_float32(flow_t1, target, "flow_t1")
    raw_mask0 = _to_device_float32(mask0, target, "mask0")
    raw_mask1 = _to_device_float32(mask1, target, "mask1")
    _validate_nchw(raw_flow0, "flow_t0", channels=2, batch=image0.shape[0])
    low_resolution = tuple(raw_flow0.shape[-2:])
    _validate_nchw(
        raw_flow1,
        "flow_t1",
        channels=2,
        batch=image0.shape[0],
        spatial=low_resolution,
    )
    _validate_nchw(
        raw_mask0,
        "mask0",
        channels=1,
        batch=image0.shape[0],
        spatial=low_resolution,
    )
    _validate_nchw(
        raw_mask1,
        "mask1",
        channels=1,
        batch=image0.shape[0],
        spatial=low_resolution,
    )
    _validate_unit_interval(raw_mask0, "mask0")
    _validate_unit_interval(raw_mask1, "mask1")

    original_size = tuple(image0.shape[-2:])
    resized_flow0 = resize_backward_flow(
        raw_flow0,
        original_size,
        network_size,
        align_corners=align_corners,
        device=target,
    )
    resized_flow1 = resize_backward_flow(
        raw_flow1,
        original_size,
        network_size,
        align_corners=align_corners,
        device=target,
    )
    resized_mask0 = resize_mask(
        raw_mask0, original_size, align_corners=align_corners, device=target
    )
    resized_mask1 = resize_mask(
        raw_mask1, original_size, align_corners=align_corners, device=target
    )

    warp0 = backward_warp(
        image0,
        resized_flow0,
        align_corners=align_corners,
        padding_mode=padding_mode,
        device=target,
    )
    warp1 = backward_warp(
        image1,
        resized_flow1,
        align_corners=align_corners,
        padding_mode=padding_mode,
        device=target,
    )
    if mask0_role == "warp0_weight":
        warp_blend = resized_mask0 * warp0 + (1.0 - resized_mask0) * warp1
    else:
        warp_blend = resized_mask0 * warp1 + (1.0 - resized_mask0) * warp0

    prediction = resized_mask1 * image1 + (1.0 - resized_mask1) * warp_blend
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("prediction contains NaN or infinity")

    return ReconstructionResult(
        flow_t0=resized_flow0,
        flow_t1=resized_flow1,
        mask0=resized_mask0,
        mask1=resized_mask1,
        warp0=warp0,
        warp1=warp1,
        warp_blend=warp_blend,
        prediction=prediction,
    )


_PACK_FIELD_CHANNELS: tuple[tuple[str, int], ...] = (
    ("flow_t0", 2),
    ("flow_t1", 2),
    ("mask0", 1),
    ("mask1", 1),
    ("warp0", 3),
    ("warp1", 3),
    ("warp_blend", 3),
    ("prediction", 3),
)


def pack_reconstruction_to_cpu(result: ReconstructionResult) -> ReconstructionResult:
    """Transfer every reconstruction field to CPU in one packed copy.

    All fields share batch and spatial shape, so concatenating them into one
    tensor turns eight device-to-host transfers into a single contiguous copy.
    """

    packed = torch.cat(
        [getattr(result, name) for name, _ in _PACK_FIELD_CHANNELS], dim=1
    )
    packed = packed.detach().to(device="cpu", dtype=torch.float32)
    fields: dict[str, torch.Tensor] = {}
    offset = 0
    for name, channels in _PACK_FIELD_CHANNELS:
        fields[name] = packed[:, offset : offset + channels]
        offset += channels
    return ReconstructionResult(**fields)


# A short, discoverable alias for callers that already know the target is t=0.5.
reconstruct = reconstruct_midpoint
