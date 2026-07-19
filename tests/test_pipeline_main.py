from __future__ import annotations

import json

import numpy as np
from PIL import Image

from vfi_hard_miner.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    RuntimeConfig,
    ThresholdConfig,
)
from vfi_hard_miner.manifest import read_jsonl
from vfi_hard_miner.pipeline import build_run_index, run_main_stage


def _frames(root):
    for index, value in enumerate((0, 64, 128, 192, 255), start=1):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = value
        Image.fromarray(image).save(root / f"01{index:05d}.png")


def test_spawned_main_stage_runs_index_to_manifest(tmp_path):
    data_root = tmp_path / "game"
    data_root.mkdir()
    _frames(data_root)
    config = AppConfig(
        data=DataConfig(root=str(data_root)),
        model=ModelConfig(
            factory="vfi_hard_miner.mock_model:create_model",
            input_height=64,
            input_width=64,
            batch_size=2,
            factory_kwargs={
                "output_scale": 2,
                "endpoint_copy_box": [0.25, 0.25, 0.75, 0.75],
            },
        ),
        thresholds=ThresholdConfig(
            wrong_reject_below=0.10,
            wrong_accept_at=0.20,
            solvable_reject_below=0.20,
            solvable_accept_at=0.50,
            missing_metrics_to_review=False,
        ),
        runtime=RuntimeConfig(
            backend="cpu",
            devices=(0,),
            workers=1,
            chunk_triplets=2,
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
        ),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(json.loads(config.canonical_json()), indent=2), encoding="utf-8"
    )
    index = build_run_index(config)
    summary = run_main_stage(config_path)
    assert index.triplets == 3
    assert summary.records == 3
    assert summary.counts == {"done": 2, "failed": 0, "pending": 0, "running": 0}
    records = list(read_jsonl(summary.manifest_path))
    assert {record["status"] for record in records} == {"accept"}
    assert all("endpoint_copy" in record["reasons"] for record in records)
