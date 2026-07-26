"""Deterministic reconstruction for midpoint frame interpolation.

The production model returns backward flows and two already-sigmoided masks.
Reconstruction runs in float32 on a caller-selected device: the CPU path is
the calibrated numerical reference, and accelerator devices (CUDA/NPU) may be
used for throughput when their ``grid_sample`` support has been probed.  With
``device=None`` or ``device="cpu"`` every operation is identical to the
original CPU reference implementation.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
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
    (see ``pack_reconstruction_to_cpu`` / ``pack_tier1_to_cpu``).  A tier-1
    partial result carries ``None`` for the warp/mask fields until tier-2 is
    materialized and merged back (see ``Tier2Residue`` / ``merge_tier2``).
    """

    flow_t0: torch.Tensor
    flow_t1: torch.Tensor
    mask0: torch.Tensor | None
    mask1: torch.Tensor | None
    warp0: torch.Tensor | None
    warp1: torch.Tensor | None
    warp_blend: torch.Tensor | None
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
    validate: bool = True,
) -> None:
    """Shape/dtype checks are unconditional; ``validate=False`` skips the
    device-synchronizing ``isfinite`` scan (production hot path)."""

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
    if validate and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or infinity")


def _validate_unit_interval(
    tensor: torch.Tensor,
    name: str,
    tolerance: float = 1e-6,
    *,
    validate: bool = True,
) -> None:
    if not validate:
        return
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
    validate: bool = True,
) -> torch.Tensor:
    """Resize backward flow and convert network-input pixels to output pixels.

    Flow values are defined in the coordinate system of the fixed network
    input, regardless of the lower spatial resolution at which the model emits
    them.  Consequently x/y are scaled by ``W_out/W_network`` and
    ``H_out/H_network`` after interpolation.
    """

    target = _resolve_device(device)
    flow_device = _to_device_float32(flow, target, "flow")
    _validate_nchw(flow_device, "flow", channels=2, validate=validate)
    out_h, out_w = _validate_hw(output_size, "output_size")
    net_h, net_w = _validate_hw(network_size, "network_size")

    # F.interpolate allocates a fresh tensor, so the in-place scaling below is
    # safe without a clone.
    resized = F.interpolate(
        flow_device,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=align_corners,
    )
    resized[:, 0].mul_(out_w / net_w)
    resized[:, 1].mul_(out_h / net_h)
    if validate and not bool(torch.isfinite(resized).all()):
        raise ValueError("resized flow contains NaN or infinity")
    return resized


def resize_mask(
    mask: torch.Tensor,
    output_size: tuple[int, int],
    *,
    align_corners: bool = False,
    device: DeviceLike = None,
    validate: bool = True,
) -> torch.Tensor:
    """Resize an already-sigmoided mask without changing its values otherwise."""

    target = _resolve_device(device)
    mask_device = _to_device_float32(mask, target, "mask")
    _validate_nchw(mask_device, "mask", channels=1, validate=validate)
    _validate_unit_interval(mask_device, "mask", validate=validate)
    out_h, out_w = _validate_hw(output_size, "output_size")
    resized = F.interpolate(
        mask_device,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=align_corners,
    )
    _validate_unit_interval(resized, "resized mask", tolerance=2e-6, validate=validate)
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
    validate: bool = True,
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
    _validate_nchw(image_device, "image", channels=3, validate=validate)
    _validate_nchw(
        flow_device,
        "backward_flow",
        channels=2,
        batch=image_device.shape[0],
        spatial=tuple(image_device.shape[-2:]),
        validate=validate,
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
    if validate and not bool(torch.isfinite(warped).all()):
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
    validate: bool = True,
) -> ReconstructionResult:
    """Reconstruct the midpoint prediction under the fixed model contract.

    ``device=None`` or ``"cpu"`` runs the calibrated CPU reference path; any
    other device runs the same operations there (result tensors stay on that
    device until the caller transfers them).

    ``validate=False`` skips the device-synchronizing NaN/range scans for the
    production hot path (shape/dtype checks still run); the scoring stage
    remains the NaN safety net.  Results are bitwise identical either way.
    """

    if mask0_role not in {"warp0_weight", "warp1_weight"}:
        raise ValueError(
            "mask0_role must be 'warp0_weight' or 'warp1_weight'; "
            f"got {mask0_role!r}"
        )

    with torch.inference_mode():
        target = _resolve_device(device)
        image0 = _to_device_float32(img0, target, "img0")
        image1 = _to_device_float32(img1, target, "img1")
        _validate_nchw(image0, "img0", channels=3, validate=validate)
        _validate_nchw(
            image1,
            "img1",
            channels=3,
            batch=image0.shape[0],
            spatial=tuple(image0.shape[-2:]),
            validate=validate,
        )
        _validate_unit_interval(image0, "img0", validate=validate)
        _validate_unit_interval(image1, "img1", validate=validate)

        raw_flow0 = _to_device_float32(flow_t0, target, "flow_t0")
        raw_flow1 = _to_device_float32(flow_t1, target, "flow_t1")
        raw_mask0 = _to_device_float32(mask0, target, "mask0")
        raw_mask1 = _to_device_float32(mask1, target, "mask1")
        _validate_nchw(
            raw_flow0, "flow_t0", channels=2, batch=image0.shape[0], validate=validate
        )
        low_resolution = tuple(raw_flow0.shape[-2:])
        _validate_nchw(
            raw_flow1,
            "flow_t1",
            channels=2,
            batch=image0.shape[0],
            spatial=low_resolution,
            validate=validate,
        )
        _validate_nchw(
            raw_mask0,
            "mask0",
            channels=1,
            batch=image0.shape[0],
            spatial=low_resolution,
            validate=validate,
        )
        _validate_nchw(
            raw_mask1,
            "mask1",
            channels=1,
            batch=image0.shape[0],
            spatial=low_resolution,
            validate=validate,
        )
        _validate_unit_interval(raw_mask0, "mask0", validate=validate)
        _validate_unit_interval(raw_mask1, "mask1", validate=validate)

        original_size = tuple(image0.shape[-2:])
        resized_flow0 = resize_backward_flow(
            raw_flow0,
            original_size,
            network_size,
            align_corners=align_corners,
            device=target,
            validate=validate,
        )
        resized_flow1 = resize_backward_flow(
            raw_flow1,
            original_size,
            network_size,
            align_corners=align_corners,
            device=target,
            validate=validate,
        )
        resized_mask0 = resize_mask(
            raw_mask0,
            original_size,
            align_corners=align_corners,
            device=target,
            validate=validate,
        )
        resized_mask1 = resize_mask(
            raw_mask1,
            original_size,
            align_corners=align_corners,
            device=target,
            validate=validate,
        )

        warp0 = backward_warp(
            image0,
            resized_flow0,
            align_corners=align_corners,
            padding_mode=padding_mode,
            device=target,
            validate=validate,
        )
        warp1 = backward_warp(
            image1,
            resized_flow1,
            align_corners=align_corners,
            padding_mode=padding_mode,
            device=target,
            validate=validate,
        )
        if mask0_role == "warp0_weight":
            warp_blend = resized_mask0 * warp0 + (1.0 - resized_mask0) * warp1
        else:
            warp_blend = resized_mask0 * warp1 + (1.0 - resized_mask0) * warp0

        prediction = resized_mask1 * image1 + (1.0 - resized_mask1) * warp_blend
        if validate and not bool(torch.isfinite(prediction).all()):
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
RECONSTRUCTION_CHANNELS = sum(channels for _, channels in _PACK_FIELD_CHANNELS)

# Two-tier transfer: tier-1 (prediction + flows) is needed for every sample
# (motion gates + base scoring), tier-2 (warps + masks) only for samples that
# pass fast-reject and reach diagnose_sample.
_TIER1_FIELDS: tuple[tuple[str, int], ...] = (
    ("flow_t0", 2),
    ("flow_t1", 2),
    ("prediction", 3),
)
_TIER2_FIELDS: tuple[tuple[str, int], ...] = (
    ("mask0", 1),
    ("mask1", 1),
    ("warp0", 3),
    ("warp1", 3),
    ("warp_blend", 3),
)
TIER1_CHANNELS = sum(channels for _, channels in _TIER1_FIELDS)
TIER2_CHANNELS = sum(channels for _, channels in _TIER2_FIELDS)


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


class Tier2Residue:
    """Tier-2 channels (warps + masks) held back from the main D2H transfer.

    The tier-2 payload stays resident (on device as one packed tensor, or on
    CPU as separate fields) until a scoring thread decides the sample passed
    fast-reject.  ``materialize`` transfers it at most once per microbatch;
    every candidate in the microbatch shares the cached CPU slices.
    """

    __slots__ = ("_packed", "_fields", "_cached", "_lock")

    def __init__(
        self,
        packed: torch.Tensor | None = None,
        fields: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if (packed is None) == (fields is None):
            raise ValueError("Tier2Residue requires exactly one of packed or fields")
        self._packed = packed
        self._fields = dict(fields) if fields is not None else None
        self._cached: dict[str, torch.Tensor] | None = None
        # Serializes concurrent D2H from scoring threads; torch_npu's transfer
        # path may not tolerate overlapping host-bound copies.
        self._lock = threading.Lock()

    @property
    def device_bytes(self) -> int:
        if self._packed is not None:
            return int(self._packed.numel()) * int(self._packed.element_size())
        if self._fields is None:
            # Device path after materialize: the packed tensor was released.
            return 0
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self._fields.values()
        )

    def materialize(self) -> dict[str, torch.Tensor]:
        """Transfer (or slice) tier-2 to CPU once; later calls reuse the cache."""

        with self._lock:
            if self._cached is None:
                if self._packed is not None:
                    packed = self._packed.detach().to(
                        device="cpu", dtype=torch.float32
                    )
                    # The device copy now lives on CPU; drop the device
                    # tensor so candidate microbatches stop occupying NPU
                    # memory for the rest of their diagnosis window.
                    self._packed = None
                    fields: dict[str, torch.Tensor] = {}
                    offset = 0
                    for name, channels in _TIER2_FIELDS:
                        fields[name] = packed[:, offset : offset + channels]
                        offset += channels
                else:
                    fields = {
                        name: tensor.detach().to(device="cpu", dtype=torch.float32)
                        for name, tensor in self._fields.items()
                    }
                self._cached = fields
            return self._cached


def pack_tier1_to_cpu(
    result: ReconstructionResult,
) -> tuple[ReconstructionResult, Tier2Residue]:
    """Transfer tier-1 (prediction + flows, 7ch) to CPU; hold tier-2 back.

    Device path: tier-1 leaves in one packed copy and tier-2 is concatenated
    into a single device-resident tensor inside the returned ``Tier2Residue``.
    CPU path: no copies at all — tier-1 fields are referenced directly and the
    residue keeps the original field tensors, so ``materialize`` degrades to
    identity transfers.  Sliced values are bitwise identical to the legacy
    18-channel pack either way.
    """

    if result.prediction.device.type != "cpu":
        tier1 = torch.cat(
            [getattr(result, name) for name, _ in _TIER1_FIELDS], dim=1
        )
        tier1 = tier1.detach().to(device="cpu", dtype=torch.float32)
        tier1_fields: dict[str, torch.Tensor] = {}
        offset = 0
        for name, channels in _TIER1_FIELDS:
            tier1_fields[name] = tier1[:, offset : offset + channels]
            offset += channels
        tier2 = torch.cat(
            [getattr(result, name) for name, _ in _TIER2_FIELDS], dim=1
        )
        residue = Tier2Residue(packed=tier2)
    else:
        tier1_fields = {name: getattr(result, name) for name, _ in _TIER1_FIELDS}
        residue = Tier2Residue(
            fields={name: getattr(result, name) for name, _ in _TIER2_FIELDS}
        )
    partial = ReconstructionResult(
        flow_t0=tier1_fields["flow_t0"],
        flow_t1=tier1_fields["flow_t1"],
        mask0=None,
        mask1=None,
        warp0=None,
        warp1=None,
        warp_blend=None,
        prediction=tier1_fields["prediction"],
    )
    return partial, residue


def pack_prediction_to_cpu(result: ReconstructionResult) -> ReconstructionResult:
    """Transfer only the prediction (3ch) to CPU; every other field is None.

    Teacher scoring consumes the prediction alone, so the flows, warps and
    masks never need to be concatenated on device nor cross the device
    boundary, and no ``Tier2Residue`` is created.
    """

    if result.prediction.device.type != "cpu":
        prediction = result.prediction.detach().to(
            device="cpu", dtype=torch.float32
        )
    else:
        prediction = result.prediction
    return ReconstructionResult(
        flow_t0=None,
        flow_t1=None,
        mask0=None,
        mask1=None,
        warp0=None,
        warp1=None,
        warp_blend=None,
        prediction=prediction,
    )


def merge_tier2(
    partial: ReconstructionResult, tier2: Mapping[str, torch.Tensor]
) -> ReconstructionResult:
    """Reassemble a full result from a tier-1 partial and materialized tier-2."""

    return ReconstructionResult(
        flow_t0=partial.flow_t0,
        flow_t1=partial.flow_t1,
        mask0=tier2["mask0"],
        mask1=tier2["mask1"],
        warp0=tier2["warp0"],
        warp1=tier2["warp1"],
        warp_blend=tier2["warp_blend"],
        prediction=partial.prediction,
    )


# A short, discoverable alias for callers that already know the target is t=0.5.
reconstruct = reconstruct_midpoint
