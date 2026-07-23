from __future__ import annotations

import numpy as np
import torch

import vfi_hard_miner.diagnostics as diagnostics_module
from vfi_hard_miner.config import AppConfig, DataConfig, ModelConfig, RuntimeConfig
from vfi_hard_miner.model_adapter import ModelOutputs
from vfi_hard_miner.reconstruction import ReconstructionResult


class _Adapter:
    def infer(self, img0, img1):
        batch = img0.shape[0]
        return ModelOutputs(
            torch.zeros((batch, 2, 2, 2)),
            torch.zeros((batch, 2, 2, 2)),
            torch.zeros((batch, 1, 2, 2)),
            torch.zeros((batch, 1, 2, 2)),
        )


def _item(index):
    image = np.zeros((6, 8, 3), dtype=np.float32)
    return ({"sample_id": f"sample-{index}"}, image, image, image)


def test_diagnostic_large_batch_uses_memory_bounded_microbatches(
    tmp_path,
    monkeypatch,
    capsys,
):
    config = AppConfig(
        data=DataConfig(root=str(tmp_path)),
        model=ModelConfig(
            factory="unused:factory",
            input_height=6,
            input_width=8,
            batch_size=64,
        ),
        runtime=RuntimeConfig(
            postproc_workers=2,
            postproc_buffer_mb=1,
            warmup_batches=0,
        ),
    )
    items = [_item(index) for index in range(65)]
    batches = [items[:64], items[64:]]
    reconstructed_sizes = []

    def decoded(*args, **kwargs):
        yield from batches

    def reconstruct(img0, img1, outputs, **kwargs):
        batch = img0.shape[0]
        reconstructed_sizes.append(batch)
        flow = torch.zeros((batch, 2, 2, 2))
        mask = torch.zeros((batch, 1, 2, 2))
        image = torch.zeros((batch, 3, 2, 2))
        return ReconstructionResult(
            flow_t0=flow,
            flow_t1=flow,
            mask0=mask,
            mask1=mask,
            warp0=image,
            warp1=image,
            warp_blend=image,
            prediction=image,
        )

    def finish(batch, reconstructed, **kwargs):
        return [{"sample_id": item[0]["sample_id"]} for item in batch]

    monkeypatch.setattr(diagnostics_module, "execution_id", lambda config: "execution")
    monkeypatch.setattr(diagnostics_module, "_prefetched_diagnostic_batches", decoded)
    monkeypatch.setattr(
        diagnostics_module,
        "_reconstruction_bytes_per_sample",
        lambda items: 100 * 1024,
    )
    monkeypatch.setattr(
        diagnostics_module,
        "_postproc_microbatch_size",
        lambda items, **kwargs: min(5, len(items)),
    )
    monkeypatch.setattr(diagnostics_module, "_reconstruct_outputs", reconstruct)
    monkeypatch.setattr(diagnostics_module, "_finish_batch_diagnostics", finish)

    result = diagnostics_module.process_diagnostic_payload(
        {
            "run_hash": config.run_hash(),
            "execution_id": "execution",
            "stage": "diagnostic",
            "records": [item[0] for item in items],
        },
        adapter=_Adapter(),
        config=config,
        artifact_root=tmp_path / "artifacts",
        progress_prefix="[diagnostic-test]",
    )

    assert [record["sample_id"] for record in result] == [
        f"sample-{index}" for index in range(65)
    ]
    assert sum(reconstructed_sizes) == 65
    assert max(reconstructed_sizes) == 5
    progress = capsys.readouterr().err
    assert "inferred 65/65" in progress
    assert "scored 65/65" in progress
