from __future__ import annotations

import sys
import types

import pytest
import torch

from vfi_hard_miner.model_adapter import (
    ModelAdapter,
    load_factory,
    normalize_model_outputs,
)
from vfi_hard_miner.config import ModelConfig
from vfi_hard_miner.runtime import collect_runtime_info, parse_device


def _valid_outputs(
    *, batch: int = 2, height: int = 2, width: int = 3
) -> dict[str, torch.Tensor]:
    return {
        "flow_t0": torch.zeros((batch, 2, height, width)),
        "flow_t1": torch.zeros((batch, 2, height, width)),
        "mask0": torch.full((batch, 1, height, width), 0.5),
        "mask1": torch.full((batch, 1, height, width), 0.25),
    }


def test_normalize_outputs_accepts_mapping_and_ordered_sequence() -> None:
    mapping = _valid_outputs()
    from_mapping = normalize_model_outputs(mapping, expected_batch=2)
    from_sequence = normalize_model_outputs(
        tuple(mapping[name] for name in ("flow_t0", "flow_t1", "mask0", "mask1")),
        expected_batch=2,
    )
    assert from_mapping.flow_t0 is mapping["flow_t0"]
    assert from_sequence.mask1 is mapping["mask1"]


@pytest.mark.parametrize(
    ("modify", "error_type", "match"),
    [
        (
            lambda output: output.pop("mask1"),
            ValueError,
            "missing required keys",
        ),
        (
            lambda output: output.__setitem__("flow_t0", torch.zeros((2, 3, 2, 3))),
            ValueError,
            r"flow_t0 must have shape \[B,2,h,w\]",
        ),
        (
            lambda output: output.__setitem__("flow_t1", torch.zeros((2, 2, 1, 3))),
            ValueError,
            "share spatial shape",
        ),
        (
            lambda output: output["flow_t0"].__setitem__((0, 0, 0, 0), float("inf")),
            ValueError,
            "contains NaN or infinity",
        ),
        (
            lambda output: output["mask0"].fill_(1.01),
            ValueError,
            "sigmoid-normalized",
        ),
    ],
)
def test_normalize_outputs_rejects_contract_violations(modify, error_type, match: str) -> None:
    output = _valid_outputs()
    modify(output)
    with pytest.raises(error_type, match=match):
        normalize_model_outputs(output, expected_batch=2)


def test_sequence_must_have_exactly_four_outputs() -> None:
    with pytest.raises(ValueError, match="exactly four"):
        normalize_model_outputs([torch.zeros(1)] * 3)


def test_factory_loader_requires_module_function_and_callable(monkeypatch) -> None:
    module = types.ModuleType("temporary_factory_module")
    module.value = 123
    module.build = lambda: object()
    monkeypatch.setitem(sys.modules, module.__name__, module)

    assert load_factory("temporary_factory_module:build") is module.build
    with pytest.raises(ValueError, match="module:function"):
        load_factory("temporary_factory_module.build")
    with pytest.raises(TypeError, match="not callable"):
        load_factory("temporary_factory_module:value")


def test_adapter_factory_runs_once_resizes_inputs_and_uses_eval(monkeypatch) -> None:
    module = types.ModuleType("temporary_vfi_model")
    calls: list[tuple[int, str | None, str]] = []

    class MockModel(torch.nn.Module):
        def __init__(self, output_scale: int) -> None:
            super().__init__()
            self.output_scale = output_scale
            self.seen_shape: tuple[int, ...] | None = None

        def forward(self, img0: torch.Tensor, img1: torch.Tensor):
            assert not self.training
            assert img0.device.type == "cpu"
            assert img0.dtype == torch.float32
            assert img0.shape == img1.shape
            self.seen_shape = tuple(img0.shape)
            batch, _, height, width = img0.shape
            out_h, out_w = height // self.output_scale, width // self.output_scale
            return {
                "flow_t0": torch.zeros((batch, 2, out_h, out_w)),
                "flow_t1": torch.zeros((batch, 2, out_h, out_w)),
                "mask0": torch.full((batch, 1, out_h, out_w), 0.5),
                "mask1": torch.zeros((batch, 1, out_h, out_w)),
            }

    def build(
        output_scale: int,
        *,
        checkpoint: str | None,
        device: torch.device,
    ) -> MockModel:
        calls.append((output_scale, checkpoint, str(device)))
        return MockModel(output_scale)

    module.build = build
    monkeypatch.setitem(sys.modules, module.__name__, module)
    adapter = ModelAdapter.from_factory(
        "temporary_vfi_model:build",
        device="cpu",
        network_size=(8, 12),
        factory_kwargs={"output_scale": 4},
    )
    output = adapter.infer(
        torch.rand((2, 3, 5, 7), dtype=torch.float64),
        torch.rand((2, 3, 5, 7), dtype=torch.float64),
    )

    assert calls == [(4, None, "cpu")]
    assert adapter.model.seen_shape == (2, 3, 8, 12)
    assert output.flow_t0.shape == (2, 2, 2, 3)


def test_adapter_resizes_on_cpu_before_accelerator_transfer(monkeypatch) -> None:
    resize_devices: list[str] = []
    original_interpolate = torch.nn.functional.interpolate

    def tracked_interpolate(value, *args, **kwargs):
        resize_devices.append(value.device.type)
        return original_interpolate(value, *args, **kwargs)

    monkeypatch.setattr(
        "vfi_hard_miner.model_adapter.F.interpolate",
        tracked_interpolate,
    )

    class MetaModel:
        def __call__(self, img0: torch.Tensor, img1: torch.Tensor):
            assert img0.device.type == "meta"
            batch = img0.shape[0]
            return {
                "flow_t0": torch.empty((batch, 2, 2, 3), device=img0.device),
                "flow_t1": torch.empty((batch, 2, 2, 3), device=img0.device),
                "mask0": torch.empty((batch, 1, 2, 3), device=img0.device),
                "mask1": torch.empty((batch, 1, 2, 3), device=img0.device),
            }

    adapter = ModelAdapter(
        MetaModel(),
        device=torch.device("meta"),
        network_size=(8, 12),
        validate_values=False,
    )
    adapter.infer(torch.rand((1, 3, 16, 24)), torch.rand((1, 3, 16, 24)))

    assert resize_devices == ["cpu", "cpu"]


def test_from_config_forwards_checkpoint_kwargs_and_sequence_output_order(monkeypatch) -> None:
    module = types.ModuleType("temporary_configured_vfi_model")
    received: list[tuple[str, str, str]] = []
    emit_order = ("mask1", "flow_t0", "mask0", "flow_t1")

    class SequenceModel(torch.nn.Module):
        def forward(self, img0: torch.Tensor, img1: torch.Tensor):
            batch = img0.shape[0]
            named = {
                "flow_t0": torch.full((batch, 2, 2, 2), 1.0),
                "flow_t1": torch.full((batch, 2, 2, 2), 2.0),
                "mask0": torch.full((batch, 1, 2, 2), 0.25),
                "mask1": torch.full((batch, 1, 2, 2), 0.75),
            }
            return tuple(named[name] for name in emit_order)

    def build(*, checkpoint: str, device: torch.device, marker: str) -> SequenceModel:
        received.append((checkpoint, str(device), marker))
        return SequenceModel()

    module.build = build
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = ModelConfig(
        factory="temporary_configured_vfi_model:build",
        checkpoint="ckpts/current/example.pth",
        input_height=4,
        input_width=4,
        output_order=emit_order,
        factory_kwargs={"marker": "current"},
    )

    adapter = ModelAdapter.from_config(config, device="cpu")
    output = adapter(torch.zeros((1, 3, 4, 4)), torch.ones((1, 3, 4, 4)))

    assert received == [("ckpts/current/example.pth", "cpu", "current")]
    torch.testing.assert_close(output.flow_t0, torch.ones_like(output.flow_t0))
    torch.testing.assert_close(output.flow_t1, torch.full_like(output.flow_t1, 2.0))
    torch.testing.assert_close(output.mask0, torch.full_like(output.mask0, 0.25))
    torch.testing.assert_close(output.mask1, torch.full_like(output.mask1, 0.75))


def test_factory_reserved_kwargs_must_not_conflict(monkeypatch) -> None:
    module = types.ModuleType("temporary_reserved_kwargs_model")
    module.build = lambda **kwargs: torch.nn.Identity()
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(ValueError, match="checkpoint conflicts"):
        ModelAdapter.from_factory(
            "temporary_reserved_kwargs_model:build",
            device="cpu",
            network_size=(4, 4),
            checkpoint="expected.pth",
            factory_kwargs={"checkpoint": "different.pth"},
        )
    with pytest.raises(ValueError, match="device conflicts"):
        ModelAdapter.from_factory(
            "temporary_reserved_kwargs_model:build",
            device="cpu",
            network_size=(4, 4),
            factory_kwargs={"device": "cuda:0"},
        )


def test_adapter_rejects_unormalized_or_mismatched_inputs() -> None:
    class NeverCalled:
        def __call__(self, img0, img1):  # pragma: no cover - contract rejects first
            raise AssertionError("model should not be called")

    adapter = ModelAdapter(NeverCalled(), device=torch.device("cpu"), network_size=(4, 4))
    with pytest.raises(ValueError, match="normalized to"):
        adapter(torch.full((1, 3, 4, 4), 1.1), torch.zeros((1, 3, 4, 4)))
    with pytest.raises(ValueError, match="identical shapes"):
        adapter(torch.zeros((1, 3, 4, 4)), torch.zeros((1, 3, 5, 4)))


def test_parent_safe_runtime_calls_do_not_import_torch_npu(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)

    assert str(parse_device("npu:7")) == "npu:7"
    info = collect_runtime_info(include_npu=False, include_npu_smi=False)

    assert "torch_npu" not in sys.modules
    assert info["npu"] == {"probed": False}
