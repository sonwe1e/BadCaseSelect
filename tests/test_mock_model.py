from __future__ import annotations

import torch

from vfi_hard_miner.mock_model import create_model


def test_mock_model_contract_and_error_box():
    model = create_model(
        checkpoint=None,
        device=torch.device("cpu"),
        output_scale=2,
        endpoint_copy_box=(0.25, 0.25, 0.75, 0.75),
    )
    image = torch.zeros(2, 3, 16, 20)
    outputs = model(image, image)
    assert outputs["flow_t0"].shape == (2, 2, 8, 10)
    assert outputs["mask0"].shape == (2, 1, 8, 10)
    assert outputs["mask1"].max() == 1
    assert outputs["mask1"].min() == 0
