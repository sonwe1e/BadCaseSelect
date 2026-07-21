"""Strict adapter for user-supplied midpoint interpolation models."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .runtime import DeviceSpec, configure_device


REQUIRED_OUTPUT_NAMES = ("flow_t0", "flow_t1", "mask0", "mask1")

# Resolved once at import time so short-name lookup never hard-codes the
# package name.  __name__ is "vfi_hard_miner.model_adapter", so rpartition
# gives "vfi_hard_miner", and we append ".models".
_MODELS_PACKAGE: str = __name__.rpartition(".")[0] + ".models"


@dataclass(frozen=True)
class ModelOutputs:
    """Normalized model outputs, still resident on the inference device."""

    flow_t0: torch.Tensor
    flow_t1: torch.Tensor
    mask0: torch.Tensor
    mask1: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "flow_t0": self.flow_t0,
            "flow_t1": self.flow_t1,
            "mask0": self.mask0,
            "mask1": self.mask1,
        }


def load_factory(factory_spec: str) -> Callable[..., Any]:
    """Resolve a model factory from a short name or an explicit ``module:function`` path.

    Short name (no ``:``)
        A bare identifier like ``"rife"`` is resolved to
        ``vfi_hard_miner.models.rife:create_model``.  The file
        ``src/vfi_hard_miner/models/rife.py`` must therefore exist and
        expose a ``create_model`` function.

    Explicit path
        The original ``"module.path:function"`` syntax is unchanged.
        Dotted attribute chains (``"pkg.mod:cls.factory"``) still work.
    """

    if not isinstance(factory_spec, str) or not factory_spec.strip():
        raise ValueError(
            "model factory must be a non-empty short name or 'module:function' string"
        )
    spec = factory_spec.strip()
    if ":" not in spec:
        if not spec.isidentifier():
            raise ValueError(
                f"model short name must be a valid Python identifier, got {spec!r}; "
                f"use 'module:function' syntax for an explicit path"
            )
        spec = f"{_MODELS_PACKAGE}.{spec}:create_model"
    module_name, separator, attribute_path = spec.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(
            f"model factory must use 'module:function' syntax or a short name, got {factory_spec!r}"
        )
    try:
        value: Any = importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(f"failed to import model factory module {module_name!r}") from exc
    try:
        for attribute in attribute_path.split("."):
            if not attribute:
                raise AttributeError(attribute_path)
            value = getattr(value, attribute)
    except AttributeError as exc:
        raise ImportError(
            f"model factory attribute {attribute_path!r} was not found in {module_name!r}"
        ) from exc
    if not callable(value):
        raise TypeError(f"model factory {factory_spec!r} is not callable")
    return value


def _validated_output_order(output_order: Sequence[str]) -> tuple[str, str, str, str]:
    order = tuple(output_order)
    if len(order) != 4 or set(order) != set(REQUIRED_OUTPUT_NAMES):
        raise ValueError(
            "output_order must contain flow_t0, flow_t1, mask0, and mask1 exactly once"
        )
    return order  # type: ignore[return-value]


def _extract_outputs(
    raw_outputs: Any,
    output_order: Sequence[str],
) -> tuple[Any, Any, Any, Any]:
    if isinstance(raw_outputs, Mapping):
        missing = [name for name in REQUIRED_OUTPUT_NAMES if name not in raw_outputs]
        if missing:
            raise ValueError(f"model output mapping is missing required keys: {missing}")
        return tuple(raw_outputs[name] for name in REQUIRED_OUTPUT_NAMES)  # type: ignore[return-value]
    if isinstance(raw_outputs, Sequence) and not isinstance(raw_outputs, (str, bytes)):
        if len(raw_outputs) != 4:
            raise ValueError(
                f"model output sequence must contain exactly four tensors, got {len(raw_outputs)}"
            )
        named_outputs = dict(zip(_validated_output_order(output_order), raw_outputs, strict=True))
        return tuple(named_outputs[name] for name in REQUIRED_OUTPUT_NAMES)  # type: ignore[return-value]
    raise TypeError(
        "model must return a mapping with flow_t0/flow_t1/mask0/mask1 "
        "or a four-element sequence"
    )


def _validate_output_tensor(
    value: Any,
    name: str,
    *,
    channels: int,
    expected_batch: int | None,
    expected_spatial: tuple[int, int] | None,
    expected_device: torch.device | None,
    validate_values: bool,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"model output {name} must be a torch.Tensor")
    if value.ndim != 4 or value.shape[1] != channels:
        raise ValueError(
            f"model output {name} must have shape [B,{channels},h,w], got {tuple(value.shape)}"
        )
    if value.shape[0] <= 0 or value.shape[2] <= 0 or value.shape[3] <= 0:
        raise ValueError(f"model output {name} dimensions must be positive")
    if expected_batch is not None and value.shape[0] != expected_batch:
        raise ValueError(
            f"model output {name} batch must be {expected_batch}, got {value.shape[0]}"
        )
    if expected_spatial is not None and tuple(value.shape[-2:]) != expected_spatial:
        raise ValueError(
            f"all model outputs must share spatial shape {expected_spatial}; "
            f"{name} has {tuple(value.shape[-2:])}"
        )
    if not value.is_floating_point():
        raise TypeError(f"model output {name} must be floating point, got {value.dtype}")
    if expected_device is not None:
        same_type = value.device.type == expected_device.type
        same_index = expected_device.index is None or value.device.index == expected_device.index
        if not (same_type and same_index):
            raise ValueError(
                f"model output {name} is on {value.device}, expected {expected_device}; "
                "implicit CPU fallback is not allowed"
            )
    if validate_values and not bool(torch.isfinite(value).all()):
        raise ValueError(f"model output {name} contains NaN or infinity")
    return value


def normalize_model_outputs(
    raw_outputs: Any,
    *,
    output_order: Sequence[str] = REQUIRED_OUTPUT_NAMES,
    expected_batch: int | None = None,
    expected_device: torch.device | None = None,
    validate_values: bool = True,
    mask_tolerance: float = 1e-6,
) -> ModelOutputs:
    """Normalize and strictly validate the four-output model contract."""

    validated_order = _validated_output_order(output_order)
    raw_flow0, raw_flow1, raw_mask0, raw_mask1 = _extract_outputs(
        raw_outputs,
        validated_order,
    )
    flow0 = _validate_output_tensor(
        raw_flow0,
        "flow_t0",
        channels=2,
        expected_batch=expected_batch,
        expected_spatial=None,
        expected_device=expected_device,
        validate_values=validate_values,
    )
    spatial = tuple(flow0.shape[-2:])
    flow1 = _validate_output_tensor(
        raw_flow1,
        "flow_t1",
        channels=2,
        expected_batch=expected_batch,
        expected_spatial=spatial,
        expected_device=expected_device,
        validate_values=validate_values,
    )
    normalized_mask0 = _validate_output_tensor(
        raw_mask0,
        "mask0",
        channels=1,
        expected_batch=expected_batch,
        expected_spatial=spatial,
        expected_device=expected_device,
        validate_values=validate_values,
    )
    normalized_mask1 = _validate_output_tensor(
        raw_mask1,
        "mask1",
        channels=1,
        expected_batch=expected_batch,
        expected_spatial=spatial,
        expected_device=expected_device,
        validate_values=validate_values,
    )
    if validate_values:
        for name, mask in (("mask0", normalized_mask0), ("mask1", normalized_mask1)):
            minimum = float(mask.amin())
            maximum = float(mask.amax())
            if minimum < -mask_tolerance or maximum > 1.0 + mask_tolerance:
                raise ValueError(
                    f"model output {name} must already be sigmoid-normalized to [0,1]; "
                    f"observed [{minimum:.8g},{maximum:.8g}]"
                )
    return ModelOutputs(flow0, flow1, normalized_mask0, normalized_mask1)


def _validate_model_input(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [B,3,H,W], got {tuple(value.shape)}")
    if value.shape[0] <= 0 or value.shape[2] <= 0 or value.shape[3] <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinity")
    minimum = float(value.amin())
    maximum = float(value.amax())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(
            f"{name} must be normalized to [0,1], observed [{minimum:.8g},{maximum:.8g}]"
        )
    return value


def _validate_network_size(network_size: tuple[int, int]) -> tuple[int, int]:
    if len(network_size) != 2:
        raise ValueError("network_size must contain (height, width)")
    height, width = int(network_size[0]), int(network_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"network_size dimensions must be positive, got {(height, width)}")
    return height, width


class ModelAdapter:
    """Move/resize inputs and enforce the model's public output contract."""

    def __init__(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], Any],
        *,
        device: torch.device,
        network_size: tuple[int, int],
        output_order: Sequence[str] = REQUIRED_OUTPUT_NAMES,
        validate_values: bool = True,
    ) -> None:
        if not callable(model):
            raise TypeError("model returned by the factory must be callable")
        self.device = torch.device(device)
        self.network_size = _validate_network_size(network_size)
        self.output_order = _validated_output_order(output_order)
        self.validate_values = bool(validate_values)

        moved_model = model
        to_method = getattr(moved_model, "to", None)
        if callable(to_method):
            maybe_moved = to_method(self.device)
            if maybe_moved is not None:
                moved_model = maybe_moved
        eval_method = getattr(moved_model, "eval", None)
        if callable(eval_method):
            eval_method()
        self.model = moved_model

    @classmethod
    def from_factory(
        cls,
        factory_spec: str,
        *,
        device: str | DeviceSpec | torch.device,
        network_size: tuple[int, int],
        checkpoint: str | None = None,
        factory_kwargs: Mapping[str, Any] | None = None,
        output_order: Sequence[str] = REQUIRED_OUTPUT_NAMES,
        validate_values: bool = True,
    ) -> "ModelAdapter":
        """Create an adapter after selecting its worker-local device.

        For ``npu:N`` this method imports ``torch_npu``.  Call it inside a
        spawned worker, not in the parent scheduler.
        """

        configured_device = configure_device(device)
        factory = load_factory(factory_spec)
        call_kwargs = dict(factory_kwargs or {})
        if "checkpoint" in call_kwargs and call_kwargs["checkpoint"] != checkpoint:
            raise ValueError("checkpoint conflicts with factory_kwargs['checkpoint']")
        call_kwargs["checkpoint"] = checkpoint
        if "device" in call_kwargs and str(call_kwargs["device"]) != str(configured_device):
            raise ValueError("configured device conflicts with factory_kwargs['device']")
        call_kwargs["device"] = configured_device
        try:
            model = factory(**call_kwargs)
        except Exception as exc:
            raise RuntimeError(f"model factory {factory_spec!r} failed") from exc
        return cls(
            model,
            device=configured_device,
            network_size=network_size,
            output_order=output_order,
            validate_values=validate_values,
        )

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        device: str | DeviceSpec | torch.device,
        validate_values: bool = True,
    ) -> "ModelAdapter":
        """Create an adapter from a ``ModelConfig``-compatible object.

        The object must expose ``factory``, ``input_height`` and
        ``input_width``.  ``checkpoint``, ``factory_kwargs`` and
        ``output_order`` are forwarded when present.  Keeping this structural
        avoids a config-module dependency and permits the same adapter for a
        teacher configuration.
        """

        try:
            factory_spec = config.factory
            network_size = (config.input_height, config.input_width)
        except AttributeError as exc:
            raise TypeError(
                "model config must expose factory, input_height, and input_width"
            ) from exc
        return cls.from_factory(
            factory_spec,
            device=device,
            network_size=network_size,
            checkpoint=getattr(config, "checkpoint", None),
            factory_kwargs=getattr(config, "factory_kwargs", None),
            output_order=getattr(config, "output_order", REQUIRED_OUTPUT_NAMES),
            validate_values=validate_values,
        )

    def _prepare_inputs(
        self, img0: torch.Tensor, img1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image0 = _validate_model_input(img0, "img0")
        image1 = _validate_model_input(img1, "img1")
        if image0.shape != image1.shape:
            raise ValueError(
                f"img0 and img1 must have identical shapes, got {tuple(image0.shape)} "
                f"and {tuple(image1.shape)}"
            )
        # Resize before H2D.  Source frames are often much larger than the
        # fixed network input, so transferring them first wastes accelerator
        # bandwidth and memory.  Keeping this stage on CPU also makes the
        # amount copied to CUDA/NPU independent of the original resolution.
        image0 = image0.detach().to(device="cpu", dtype=torch.float32)
        image1 = image1.detach().to(device="cpu", dtype=torch.float32)
        if tuple(image0.shape[-2:]) != self.network_size:
            image0 = F.interpolate(
                image0,
                size=self.network_size,
                mode="bilinear",
                align_corners=False,
            )
            image1 = F.interpolate(
                image1,
                size=self.network_size,
                mode="bilinear",
                align_corners=False,
            )
        image0 = image0.to(self.device, non_blocking=True)
        image1 = image1.to(self.device, non_blocking=True)
        return image0, image1

    def infer(self, img0: torch.Tensor, img1: torch.Tensor) -> ModelOutputs:
        prepared0, prepared1 = self._prepare_inputs(img0, img1)
        with torch.inference_mode():
            raw_outputs = self.model(prepared0, prepared1)
        return normalize_model_outputs(
            raw_outputs,
            output_order=self.output_order,
            expected_batch=prepared0.shape[0],
            expected_device=self.device,
            validate_values=self.validate_values,
        )

    __call__ = infer


def load_model_adapter(
    factory_spec: str,
    *,
    device: str | DeviceSpec | torch.device,
    network_size: tuple[int, int],
    checkpoint: str | None = None,
    factory_kwargs: Mapping[str, Any] | None = None,
    output_order: Sequence[str] = REQUIRED_OUTPUT_NAMES,
    validate_values: bool = True,
) -> ModelAdapter:
    """Functional wrapper around :meth:`ModelAdapter.from_factory`."""

    return ModelAdapter.from_factory(
        factory_spec,
        device=device,
        network_size=network_size,
        checkpoint=checkpoint,
        factory_kwargs=factory_kwargs,
        output_order=output_order,
        validate_values=validate_values,
    )
