from __future__ import annotations

import json

import numpy as np
from PIL import Image

from vfi_hard_miner.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
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


def test_per_video_main_stage_materializes_completed_video(tmp_path):
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
        output=OutputConfig(
            link_mode="copy",
            materialize_strategy="per_video",
            visualization_width=64,
        ),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(json.loads(config.canonical_json()), indent=2), encoding="utf-8"
    )

    build_run_index(config)
    summary = run_main_stage(config_path)

    assert summary.materialization is not None
    assert summary.materialization.videos == 1
    assert summary.materialization.segments == 1
    assert summary.materialization.frames == 5
    assert summary.materialization.link_counts["copy"] == 5
    staged = summary.materialization.staging_path
    assert staged is not None
    segment_directories = list((staged / "01").glob("segment_*"))
    assert len(segment_directories) == 1
    staged_frames = list(segment_directories[0].glob("*.png"))
    assert len(staged_frames) == 5
    staged_frames[0].unlink()

    resumed = run_main_stage(config_path)
    assert resumed.materialization is not None
    assert resumed.materialization.videos == 1
    assert resumed.materialization.frames == 5
    assert resumed.materialization.link_counts["copy"] == 5
    assert len(list((staged / "01").glob("segment_*/*.png"))) == 5
