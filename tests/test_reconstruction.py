from __future__ import annotations

import pytest
import torch

from vfi_hard_miner.reconstruction import (
    backward_warp,
    reconstruct_midpoint,
    resize_backward_flow,
)


def _constant_image(value: float, *, height: int = 2, width: int = 3) -> torch.Tensor:
    return torch.full((1, 3, height, width), value, dtype=torch.float32)


def _zero_flow(*, height: int = 2, width: int = 3) -> torch.Tensor:
    return torch.zeros((1, 2, height, width), dtype=torch.float32)


def _mask(value: float, *, height: int = 2, width: int = 3) -> torch.Tensor:
    return torch.full((1, 1, height, width), value, dtype=torch.float32)


def test_resize_flow_uses_network_input_pixel_units() -> None:
    flow = torch.empty((1, 2, 1, 1), dtype=torch.float64)
    flow[:, 0] = 1.0
    flow[:, 1] = 2.0

    resized = resize_backward_flow(
        flow,
        output_size=(4, 6),
        network_size=(2, 2),
    )

    assert resized.device.type == "cpu"
    assert resized.dtype == torch.float32
    torch.testing.assert_close(resized[:, 0], torch.full((1, 4, 6), 3.0))
    torch.testing.assert_close(resized[:, 1], torch.full((1, 4, 6), 4.0))


@pytest.mark.parametrize("align_corners", [False, True])
def test_backward_warp_identity(align_corners: bool) -> None:
    image = torch.linspace(0.0, 1.0, 3 * 4 * 5).reshape(1, 3, 4, 5)
    warped = backward_warp(
        image,
        torch.zeros((1, 2, 4, 5)),
        align_corners=align_corners,
        padding_mode="zeros",
    )
    torch.testing.assert_close(warped, image, atol=2e-6, rtol=0.0)


def test_backward_flow_is_target_to_source_pixel_displacement() -> None:
    image = torch.zeros((1, 3, 3, 5))
    image[:, :, 1, 1] = 1.0
    flow = torch.zeros((1, 2, 3, 5))
    flow[:, 0] = -1.0

    warped = backward_warp(image, flow, padding_mode="zeros")

    expected = torch.zeros_like(image)
    expected[:, :, 1, 2] = 1.0
    torch.testing.assert_close(warped, expected, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    ("role", "expected"),
    [("warp0_weight", 0.65), ("warp1_weight", 0.35)],
)
def test_mask0_role_is_global_and_explicit(role: str, expected: float) -> None:
    result = reconstruct_midpoint(
        _constant_image(0.2),
        _constant_image(0.8),
        _zero_flow(),
        _zero_flow(),
        _mask(0.25),
        _mask(0.0),
        network_size=(2, 3),
        mask0_role=role,  # type: ignore[arg-type]
        padding_mode="border",
    )

    torch.testing.assert_close(result.warp_blend, _constant_image(expected), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(result.prediction, result.warp_blend)


def test_mask1_residual_copies_img1_with_configured_weight() -> None:
    result = reconstruct_midpoint(
        _constant_image(0.2),
        _constant_image(0.8),
        _zero_flow(),
        _zero_flow(),
        _mask(1.0),
        _mask(0.25),
        network_size=(2, 3),
        mask0_role="warp0_weight",
    )

    torch.testing.assert_close(result.warp_blend, _constant_image(0.2))
    torch.testing.assert_close(result.prediction, _constant_image(0.35), atol=1e-6, rtol=0.0)


def test_reconstruction_converts_all_products_to_cpu_float32() -> None:
    result = reconstruct_midpoint(
        _constant_image(0.1).double(),
        _constant_image(0.9).double(),
        _zero_flow().double(),
        _zero_flow().double(),
        _mask(0.5).double(),
        _mask(0.0).double(),
        network_size=(2, 3),
        mask0_role="warp0_weight",
    )
    for value in result.__dict__.values():
        assert value.device.type == "cpu"
        assert value.dtype == torch.float32


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda values: values.__setitem__(2, _mask(1.1)), "mask0 must be in"),
        (
            lambda values: values.__setitem__(3, torch.zeros((1, 1, 1, 3))),
            "mask1 spatial shape",
        ),
        (
            lambda values: values[0].__setitem__((0, 0, 0, 0), float("nan")),
            "flow_t0 contains NaN",
        ),
    ],
)
def test_reconstruction_rejects_invalid_model_outputs(mutate, match: str) -> None:
    values = [_zero_flow(), _zero_flow(), _mask(0.5), _mask(0.0)]
    mutate(values)
    with pytest.raises(ValueError, match=match):
        reconstruct_midpoint(
            _constant_image(0.2),
            _constant_image(0.8),
            *values,
            network_size=(2, 3),
            mask0_role="warp0_weight",
        )

def test_reconstruction_rejects_unknown_mask_role() -> None:
    with pytest.raises(ValueError, match="mask0_role"):
        reconstruct_midpoint(
            _constant_image(0.2),
            _constant_image(0.8),
            _zero_flow(),
            _zero_flow(),
            _mask(0.5),
            _mask(0.0),
            network_size=(2, 3),
            mask0_role="choose_best_from_gt",  # type: ignore[arg-type]
        )

# ---------------------------------------------------------------------------
# device parameter: explicit CPU must match the reference; packed D2H helper
# ---------------------------------------------------------------------------

from vfi_hard_miner.reconstruction import pack_reconstruction_to_cpu  # noqa: E402


def _random_reconstruction_inputs():
    torch.manual_seed(7)
    img0 = torch.rand((2, 3, 5, 7))
    img1 = torch.rand((2, 3, 5, 7))
    flow0 = (torch.rand((2, 2, 2, 3)) - 0.5) * 2.0
    flow1 = (torch.rand((2, 2, 2, 3)) - 0.5) * 2.0
    mask0 = torch.rand((2, 1, 2, 3))
    mask1 = torch.rand((2, 1, 2, 3))
    return img0, img1, flow0, flow1, mask0, mask1


@pytest.mark.parametrize("mask0_role", ["warp0_weight", "warp1_weight"])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_explicit_cpu_device_matches_default_reference(
    mask0_role, align_corners, padding_mode
) -> None:
    inputs = _random_reconstruction_inputs()
    kwargs = dict(
        network_size=(2, 3),
        mask0_role=mask0_role,
        align_corners=align_corners,
        padding_mode=padding_mode,
    )
    reference = reconstruct_midpoint(*inputs, **kwargs)
    explicit = reconstruct_midpoint(*inputs, **kwargs, device="cpu")
    for name, tensor in reference.__dict__.items():
        torch.testing.assert_close(explicit.__dict__[name], tensor)
        assert explicit.__dict__[name].device.type == "cpu"


def test_pack_reconstruction_to_cpu_round_trips_all_fields() -> None:
    result = reconstruct_midpoint(
        *_random_reconstruction_inputs(),
        network_size=(2, 3),
        mask0_role="warp0_weight",
    )
    packed = pack_reconstruction_to_cpu(result)
    for name, tensor in result.__dict__.items():
        field = packed.__dict__[name]
        assert field.device.type == "cpu"
        assert field.dtype == torch.float32
        torch.testing.assert_close(field, tensor)
