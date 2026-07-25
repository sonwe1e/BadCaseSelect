from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import vfi_hard_miner.cgvqm_stage as stage_module
from vfi_hard_miner.cgvqm import CGVQMResult
from vfi_hard_miner.cgvqm_stage import run_cgvqm_stage
from vfi_hard_miner.config import (
    AppConfig,
    CGVQMConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    RuntimeConfig,
    ThresholdConfig,
)
from vfi_hard_miner.finalize import finalize_run
from vfi_hard_miner.manifest import read_jsonl
from vfi_hard_miner.pipeline import build_run_index, run_main_stage, stage_counts


class _FakeCGVQM:
    def __init__(self, *args, device="cpu", **kwargs):
        self.device = str(device)

    def probe(self, **kwargs) -> None:
        return None

    def score(self, distorted, reference) -> CGVQMResult:
        batch, frames, height, width, _ = np.asarray(distorted).shape
        return CGVQMResult(
            errors=np.full(batch, 60.0, dtype=np.float32),
            error_maps=np.ones(
                (batch, frames, height, width),
                dtype=np.float32,
            ),
            backend=self.device,
        )


class _ReuseOnlyCGVQM(_FakeCGVQM):
    def score(self, distorted, reference) -> CGVQMResult:
        raise AssertionError("completed CGVQM video was scored again")


def test_cgvqm_stage_grades_materializes_and_finalizes_flat_training_data(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "game"
    root.mkdir()
    for index, value in enumerate((0, 64, 128, 192, 255), start=1):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = value
        Image.fromarray(image).save(root / f"01{index:05d}.png")
    backbone = tmp_path / "r3d.pth"
    calibration = tmp_path / "cgvqm.pickle"
    backbone.write_bytes(b"fake-backbone")
    calibration.write_bytes(b"fake-calibration")
    config = AppConfig(
        data=DataConfig(root=str(root)),
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
        cgvqm=CGVQMConfig(
            enabled=True,
            backbone_checkpoint=str(backbone),
            calibration_checkpoint=str(calibration),
            backend="cpu",
            clip_frames=3,
            crop_size=32,
            candidates_per_task=2,
            b_error_at=0.1,
            a_error_at=0.5,
            temporal_persistence_at=0.2,
            spatial_overlap_at=0.1,
        ),
        thresholds=ThresholdConfig(
            wrong_reject_below=0.10,
            wrong_accept_at=0.20,
            severe_wrong_accept_at=0.20,
            solvable_reject_below=0.20,
            solvable_accept_at=0.50,
            missing_metrics_to_review=False,
        ),
        runtime=RuntimeConfig(
            backend="cpu",
            devices=(0,),
            workers=1,
            chunk_triplets=2,
            warmup_batches=0,
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
        ),
        output=OutputConfig(
            link_mode="copy",
            layout="graded_flat",
            materialize_strategy="per_video",
            visualization_width=64,
        ),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(json.loads(config.canonical_json()), indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(stage_module, "CGVQM2Scorer", _FakeCGVQM)

    build_run_index(config)
    run_main_stage(config_path)
    refined = run_cgvqm_stage(config_path)

    assert refined.refined == 3
    assert refined.materialization is not None
    assert stage_counts(config, stage="cgvqm") == {
        "done": 1,
        "failed": 0,
        "pending": 0,
        "running": 0,
    }
    staged = refined.materialization.staging_path
    assert sorted(path.name for path in (staged / "A").glob("*.png")) == [
        f"01{index:05d}.png" for index in range(1, 6)
    ]
    assert not list((staged / "A").glob("*/*"))
    monkeypatch.setattr(stage_module, "CGVQM2Scorer", _ReuseOnlyCGVQM)
    reused = run_cgvqm_stage(config_path)
    assert reused.reused_videos == 1
    assert reused.materialization is not None
    assert reused.materialization.copy_counts == refined.materialization.copy_counts

    summary = finalize_run(config_path)
    assert summary.grades == {"A": 3, "B": 0, "Review": 0, "Reject": 0}
    assert summary.grade_frames == {"A": 5, "B": 0}
    published = root / "extremely_hard_case"
    assert sorted(path.name for path in (published / "A").glob("*.png")) == [
        f"01{index:05d}.png" for index in range(1, 6)
    ]
    assert (published / "B").is_dir()
    assert not list((published / "A").glob("*/*"))
    for source in root.glob("01*.png"):
        assert (published / "A" / source.name).read_bytes() == source.read_bytes()
    manifest = list(read_jsonl(summary.manifest_path))
    assert all(record["grade"] == "A" for record in manifest)
    assert all(Path(record["visualization"]).suffix == ".jpg" for record in manifest)
    assert all(Path(record["visualization"]).is_file() for record in manifest)
    with Image.open(manifest[0]["visualization"]) as diagnostic:
        assert diagnostic.format == "JPEG"
        assert diagnostic.size == (320, 184)


def test_window_indices_pad_at_video_boundaries_without_crossing():
    assert stage_module._window_indices(3, 0, 5) == (0, 0, 0, 1, 2)
    assert stage_module._window_indices(3, 2, 5) == (0, 1, 2, 2, 2)


def test_cgvqm_batch_is_bounded_by_the_per_worker_memory_budget(tmp_path):
    config = AppConfig(
        data=DataConfig(root=str(tmp_path)),
        model=ModelConfig(factory="mock"),
        cgvqm=CGVQMConfig(
            enabled=True,
            clip_frames=16,
            crop_size=224,
            batch_size=8,
        ),
        runtime=RuntimeConfig(postproc_buffer_mb=128),
    )

    assert stage_module._cgvqm_microbatch_size(config) == 1
