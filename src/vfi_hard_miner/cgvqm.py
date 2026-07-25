"""Offline CGVQM-2 scorer with a torch-only R3D-18 backbone.

The feature-distance calculation follows the pinned IntelLabs/CGVQM source.
No runtime downloads or torchvision model construction are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


_MEAN = (0.43216, 0.394666, 0.37645)
_STD = (0.22803, 0.22145, 0.216989)
_FEATURE_CHANNELS = (3, 64, 64)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value if self.downsample is None else self.downsample(value)
        output = self.conv1(value)
        output = self.conv2(output)
        return self.relu(output + residual)


class R3D18(nn.Module):
    """R3D-18 with torchvision-compatible parameter names."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(
                3,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, 400)

    @staticmethod
    def _make_layer(
        inplanes: int, planes: int, *, blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            _BasicBlock(inplanes, planes, stride=stride)
        ]
        layers.extend(_BasicBlock(planes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.stem(value)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return self.fc(self.avgpool(output).flatten(1))

    def cgvqm2_features(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        stem = self.stem(value)
        first_block = self.layer1(stem)
        return value, stem, first_block


@dataclass(frozen=True, slots=True)
class CGVQMResult:
    errors: np.ndarray
    error_maps: np.ndarray
    backend: str


def _torch_load_weights(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older server torch
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in payload.items()
    ):
        raise ValueError(f"invalid R3D checkpoint: {path}")
    return payload


def _load_calibration(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    # The file is accepted only after the project manifest hash check.  The
    # pinned upstream format is a tuple of two tensors.
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # nosec B301 - pinned, hash-verified asset
    if (
        not isinstance(payload, (list, tuple))
        or len(payload) != 2
        or not torch.is_tensor(payload[0])
        or not torch.is_tensor(payload[1])
    ):
        raise ValueError(f"invalid CGVQM calibration checkpoint: {path}")
    weights = payload[0].detach().to(dtype=torch.float32, device="cpu")
    alpha = payload[1].detach().to(dtype=torch.float32, device="cpu")
    if tuple(weights.shape) != (1, sum(_FEATURE_CHANNELS), 1, 1, 1):
        raise ValueError(
            f"unexpected CGVQM-2 feature weights {tuple(weights.shape)}"
        )
    if alpha.numel() != 1:
        raise ValueError("CGVQM alpha must be scalar")
    if not torch.isfinite(weights).all() or not torch.isfinite(alpha).all():
        raise ValueError("CGVQM calibration contains non-finite values")
    return weights, alpha.reshape(())


def _as_video_tensor(value: Any, *, name: str) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim != 5:
        raise ValueError(f"{name} must be BxTxHxWx3 or Bx3xTxHxW")
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 4, 1, 2, 3)
    elif tensor.shape[1] != 3:
        raise ValueError(f"{name} must contain three RGB channels")
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(dtype=torch.float32) / 255.0
    else:
        tensor = tensor.to(dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    minimum = float(tensor.amin())
    maximum = float(tensor.amax())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(f"{name} must be RGB in [0,1]")
    return tensor.clamp(0.0, 1.0)


class CGVQM2Scorer:
    def __init__(
        self,
        backbone_checkpoint: str | Path,
        calibration_checkpoint: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        backbone_path = Path(backbone_checkpoint).expanduser().resolve()
        calibration_path = Path(calibration_checkpoint).expanduser().resolve()
        if not backbone_path.is_file():
            raise FileNotFoundError(backbone_path)
        if not calibration_path.is_file():
            raise FileNotFoundError(calibration_path)
        self.device = torch.device(device)
        self.model = R3D18()
        state = _torch_load_weights(backbone_path)
        self.model.load_state_dict(state, strict=True)
        feature_weights, alpha = _load_calibration(calibration_path)
        self.feature_weights = feature_weights.to(self.device).abs()
        self.alpha = alpha.to(self.device)
        self.model.to(self.device).eval()
        mean = torch.tensor(_MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = torch.tensor(_STD, dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)

    def _preprocess(self, value: Any, *, name: str) -> torch.Tensor:
        tensor = _as_video_tensor(value, name=name).to(self.device)
        return (tensor - self.mean) / self.std

    @staticmethod
    def _feature_difference(
        left: torch.Tensor,
        right: torch.Tensor,
        weights: torch.Tensor,
        *,
        output_size: Sequence[int],
        normalize_channels: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if normalize_channels:
            left = F.normalize(left, p=2.0, dim=1, eps=1e-10)
            right = F.normalize(right, p=2.0, dim=1, eps=1e-10)
        squared = (left - right).square()
        weighted = squared * weights
        errors = weighted.mean(dim=(2, 3, 4)).sum(dim=1)
        error_map = weighted.sum(dim=1, keepdim=True)
        error_map = F.interpolate(
            error_map,
            size=tuple(int(item) for item in output_size),
            mode="trilinear",
            align_corners=False,
        )
        return errors, error_map

    def score(self, distorted: Any, reference: Any) -> CGVQMResult:
        distorted_tensor = self._preprocess(distorted, name="distorted")
        reference_tensor = self._preprocess(reference, name="reference")
        if distorted_tensor.shape != reference_tensor.shape:
            raise ValueError(
                "distorted and reference videos must have the same shape"
            )
        output_size = distorted_tensor.shape[2:]
        with torch.inference_mode():
            distorted_features = self.model.cgvqm2_features(distorted_tensor)
            reference_features = self.model.cgvqm2_features(reference_tensor)
            split_weights = self.feature_weights.split(_FEATURE_CHANNELS, dim=1)
            errors = torch.zeros(
                distorted_tensor.shape[0],
                dtype=torch.float32,
                device=self.device,
            )
            error_maps = torch.zeros(
                (
                    distorted_tensor.shape[0],
                    1,
                    *output_size,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            for index, (left, right, weights) in enumerate(
                zip(distorted_features, reference_features, split_weights)
            ):
                layer_error, layer_map = self._feature_difference(
                    left,
                    right,
                    weights,
                    output_size=output_size,
                    normalize_channels=index > 0,
                )
                errors += layer_error
                error_maps += layer_map
            errors *= self.alpha
            error_maps *= self.alpha
            if not torch.isfinite(errors).all() or not torch.isfinite(error_maps).all():
                raise RuntimeError("CGVQM produced non-finite output")
            errors_cpu = errors.detach().to(device="cpu", dtype=torch.float32).numpy()
            maps_cpu = (
                error_maps[:, 0]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
            )
        return CGVQMResult(
            errors=np.asarray(errors_cpu, dtype=np.float32),
            error_maps=np.asarray(maps_cpu, dtype=np.float32),
            backend=str(self.device),
        )

    def probe(self, *, clip_frames: int = 4, crop_size: int = 64) -> None:
        zeros = torch.zeros(
            (1, clip_frames, crop_size, crop_size, 3), dtype=torch.float32
        )
        result = self.score(zeros, zeros)
        if result.errors.shape != (1,) or result.error_maps.shape != (
            1,
            clip_frames,
            crop_size,
            crop_size,
        ):
            raise RuntimeError("CGVQM probe returned an unexpected shape")


__all__ = ["CGVQM2Scorer", "CGVQMResult", "R3D18"]
