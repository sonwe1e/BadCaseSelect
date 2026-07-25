from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

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
from vfi_hard_miner.pipeline import (
    build_run_index,
    run_directory,
    run_main_stage,
    stage_counts,
)


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


class _ContentSensitiveCGVQM(_FakeCGVQM):
    """Scores derived from clip content, so misfilled slots change results."""

    def score(self, distorted, reference) -> CGVQMResult:
        distorted = np.asarray(distorted, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        batch = distorted.shape[0]
        delta = np.abs(distorted - reference).reshape(batch, -1).mean(axis=1)
        errors = np.clip(delta / 25.5, 0.0, 100.0).astype(np.float32)
        return CGVQMResult(
            errors=errors,
            error_maps=np.ones(distorted.shape[:4], dtype=np.float32),
            backend=self.device,
        )


def test_context_cache_lru_evicts_within_budget():
    cache = stage_module._ContextCache(budget_bytes=1000)
    big = np.zeros((10, 10, 3), dtype=np.float32)  # 1200 bytes alone
    cache.put("big", big, big)
    assert cache.get("big") is None

    pred = np.zeros((4, 4, 3), dtype=np.float32)  # 192 bytes
    gt = np.zeros((4, 4, 3), dtype=np.uint8)  # 48 bytes -> 240 per entry
    for name in ("a", "b", "c", "d"):
        cache.put(name, pred, gt)
    cache.put("e", pred, gt)  # forces eviction of the oldest entry
    assert cache.get("a") is None
    assert cache.get("e") is not None
    cache.clear()
    assert cache.get("e") is None


def test_context_cache_mb_validation():
    CGVQMConfig(context_cache_mb=0).validate()
    with pytest.raises(ValueError, match="context_cache_mb"):
        CGVQMConfig(context_cache_mb=-1).validate()


def _cgvqm_cache_fixture(tmp_path, *, run_name: str, context_cache_mb: int):
    data_root = tmp_path / "game"
    data_root.mkdir(exist_ok=True)
    for index, value in enumerate((0, 64, 128, 192, 255), start=1):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = value
        Image.fromarray(image).save(data_root / f"01{index:05d}.png")
    backbone = tmp_path / "r3d.pth"
    calibration = tmp_path / "cgvqm.pickle"
    backbone.write_bytes(b"fake-backbone")
    calibration.write_bytes(b"fake-calibration")
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
        cgvqm=CGVQMConfig(
            enabled=True,
            backbone_checkpoint=str(backbone),
            calibration_checkpoint=str(calibration),
            backend="cpu",
            clip_frames=3,
            crop_size=32,
            candidates_per_task=2,
            context_cache_mb=context_cache_mb,
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
            state_db=str(tmp_path / run_name / "state.sqlite3"),
            run_dir=str(tmp_path / run_name / "run"),
        ),
        output=OutputConfig(
            link_mode="copy",
            layout="graded_flat",
            materialize_strategy="per_video",
            visualization_width=64,
        ),
    )
    config_path = tmp_path / f"config-{run_name}.json"
    config_path.write_text(
        json.dumps(json.loads(config.canonical_json()), indent=2),
        encoding="utf-8",
    )
    return config_path, config


def _evidence_numbers(config):
    parts_dir = run_directory(config) / "cgvqm_parts"
    records = []
    for part in sorted(parts_dir.glob("*.jsonl")):
        records.extend(read_jsonl(part))
    numeric_fields = (
        "error",
        "clip_error",
        "center_error",
        "temporal_persistence",
        "temporal_change",
        "spatial_overlap",
    )
    return sorted(
        (
            (str(record["sample_id"]),),
            tuple(float(record[field]) for field in numeric_fields),
        )
        for record in records
    )


def test_context_cache_reduces_reconstruction_without_changing_scores(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(stage_module, "CGVQM2Scorer", _ContentSensitiveCGVQM)
    reconstruction_calls = {"cached": 0, "uncached": 0}
    real_reconstruct = stage_module._reconstruct_predictions_cpu

    def make_spy(bucket):
        def spy(*args, **kwargs):
            reconstruction_calls[bucket] += 1
            return real_reconstruct(*args, **kwargs)

        return spy

    def run_once(run_name, context_cache_mb, bucket):
        config_path, config = _cgvqm_cache_fixture(
            tmp_path, run_name=run_name, context_cache_mb=context_cache_mb
        )
        monkeypatch.setattr(
            stage_module, "_reconstruct_predictions_cpu", make_spy(bucket)
        )
        build_run_index(config)
        run_main_stage(config_path)
        summary = run_cgvqm_stage(config_path)
        return config, summary

    uncached_config, uncached = run_once("uncached", 0, "uncached")
    cached_config, cached = run_once("cached", 256, "cached")

    assert uncached.refined == cached.refined > 0
    # Chunks share context frames; the cached run must reconstruct fewer.
    assert reconstruction_calls["cached"] < reconstruction_calls["uncached"]
    assert _evidence_numbers(uncached_config) == _evidence_numbers(cached_config)


def test_concurrent_claim_loops_complete_tasks_exactly_once(tmp_path, monkeypatch):
    import threading

    monkeypatch.setattr(stage_module, "CGVQM2Scorer", _FakeCGVQM)
    config_path, config = _cgvqm_cache_fixture(
        tmp_path, run_name="concurrent", context_cache_mb=256
    )
    build_run_index(config)
    run_main_stage(config_path)
    context = stage_module._prepare_cgvqm_context(config)

    errors = []

    def work():
        try:
            stage_module._run_cgvqm_claim_loop(context)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60.0)

    assert errors == []
    summary = stage_module._finalize_cgvqm_stage(context, backend_label="cpu")
    assert summary.refined == 3
    assert summary.counts == {
        "done": 1,
        "failed": 0,
        "pending": 0,
        "running": 0,
    }
