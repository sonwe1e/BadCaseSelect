from __future__ import annotations

import pytest
import torch

from vfi_hard_miner.config import ModelConfig
from vfi_hard_miner.model_adapter import ModelAdapter
from vfi_hard_miner.reconstruction import reconstruct_midpoint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is not available")
def test_cuda_model_outputs_feed_common_cpu_reconstruction():
    config = ModelConfig(
        factory="vfi_hard_miner.mock_model:create_model",
        input_height=32,
        input_width=48,
        batch_size=2,
        factory_kwargs={"output_scale": 2},
    )
    adapter = ModelAdapter.from_config(config, device="cuda:0")
    img0 = torch.zeros(2, 3, 32, 48)
    img1 = torch.ones(2, 3, 32, 48)
    outputs = adapter.infer(img0, img1)
    assert outputs.flow_t0.is_cuda
    result = reconstruct_midpoint(
        img0,
        img1,
        outputs.flow_t0,
        outputs.flow_t1,
        outputs.mask0,
        outputs.mask1,
        network_size=(32, 48),
        mask0_role="warp0_weight",
    )
    assert result.prediction.device.type == "cpu"
    assert torch.allclose(result.prediction, torch.full_like(result.prediction, 0.5))
