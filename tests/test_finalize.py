from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import vfi_hard_miner.finalize as finalize_module
from vfi_hard_miner.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    RuntimeConfig,
    ThresholdConfig,
)
from vfi_hard_miner.finalize import finalize_run
from vfi_hard_miner.manifest import read_jsonl, write_jsonl_part
from vfi_hard_miner.pipeline import build_run_index, run_main_stage


def _completed_case(tmp_path):
    root = tmp_path / "game"
    root.mkdir()
    for index, value in enumerate((0, 64, 128, 192, 255), start=1):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = value
        Image.fromarray(image).save(root / f"01{index:05d}.png")
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
            warmup_batches=0,
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
        ),
        output=OutputConfig(link_mode="copy", visualization_width=64),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(json.loads(config.canonical_json()), indent=2), encoding="utf-8"
    )
    build_run_index(config)
    main = run_main_stage(config_path)
    return root, config, config_path, main


def test_finalize_merges_and_materializes_continuous_frames(tmp_path):
    root = tmp_path / "game"
    root.mkdir()
    for index, value in enumerate((0, 64, 128, 192, 255), start=1):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = value
        Image.fromarray(image).save(root / f"01{index:05d}.png")
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
        output=OutputConfig(link_mode="copy", visualization_width=64),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(json.loads(config.canonical_json()), indent=2), encoding="utf-8"
    )
    build_run_index(config)
    run_main_stage(config_path)
    summary = finalize_run(config_path)
    assert summary.segments == 1
    assert summary.frames == 5
    assert summary.visualizations == 3
    copied = sorted((root / "extremely_hard_case").rglob("*.png"))
    assert [path.name for path in copied] == [f"01{index:05d}.png" for index in range(1, 6)]
    assert len({path.parent for path in copied}) == 1
    manifest = list(read_jsonl(root / "hard_case_manifest.jsonl"))
    assert len(manifest) == 3
    assert all(record["selected"] for record in manifest)
    assert all(record["covered_by_segment"] for record in manifest)
    assert all(len(record["segment_output_directories"]) == 1 for record in manifest)
    output_directory = manifest[0]["segment_output_directories"][0]
    assert all(
        record["segment_output_directories"] == [output_directory]
        for record in manifest
    )
    segments = json.loads(summary.segment_path.read_text(encoding="utf-8"))
    assert segments[0]["output_directory"] == output_directory
    assert segments[0]["trainable_triplets"] == 3
    assert (root / "extremely_hard_case" / output_directory).is_dir()
    assert all(Path(record["visualization"]).is_file() for record in manifest)
    assert (root / "extremely_hard_case" / ".vfi_hard_miner_output.json").is_file()
    assert (root / ".vfi_hard_miner_current.json").is_file()


def test_finalize_does_not_mark_a_covered_review_as_a_hard_center(tmp_path):
    root = tmp_path / "game"
    root.mkdir()
    for index, value in enumerate((0, 64, 128, 192, 255), start=1):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = value
        Image.fromarray(image).save(root / f"01{index:05d}.png")
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
            warmup_batches=0,
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
        ),
        output=OutputConfig(link_mode="copy", visualization_width=64),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(json.loads(config.canonical_json()), indent=2), encoding="utf-8"
    )
    build_run_index(config)
    main = run_main_stage(config_path)
    records = list(read_jsonl(main.manifest_path))
    assert all(record["status"] == "accept" for record in records)
    review_record = next(record for record in records if record["frame_indices"] == [2, 3, 4])
    review_record["status"] = "review"
    review_record["reasons"] = ["wrongness_gray_zone"]
    write_jsonl_part(main.manifest_path, records)

    summary = finalize_run(config_path)
    manifest = list(read_jsonl(summary.manifest_path))
    middle = next(
        record for record in manifest if record["sample_id"] == review_record["sample_id"]
    )
    assert middle["covered_by_segment"] is True
    assert middle["selected"] is False
    assert middle["visualization"] is None
    assert summary.accepted_records == 2
    assert summary.visualizations == 2


def test_segment_relative_plan_prevents_stride_crossing_between_segments(tmp_path):
    frames = {}
    for index in (1, 3, 5, 7, 9, 11):
        path = tmp_path / f"01{index:05d}.png"
        path.write_bytes(str(index).encode("ascii"))
        frames[index] = path
    segments = (
        finalize_module.FrameInterval("scene::01", 1, 5),
        finalize_module.FrameInterval("scene::01", 7, 11),
    )

    mappings, directories = finalize_module._segment_materialization_plan(
        segments,
        {"scene::01": frames},
        run_hash="run-hash",
    )

    assert len(directories) == 2
    by_leaf = {}
    for source, relative in mappings:
        by_leaf.setdefault(relative.parent.as_posix(), []).append(int(source.stem[-5:]))
    assert sorted(sorted(indices) for indices in by_leaf.values()) == [
        [1, 3, 5],
        [7, 9, 11],
    ]
    assert not any({3, 5, 7}.issubset(indices) for indices in map(set, by_leaf.values()))

    retained = finalize_module._retain_segments_with_hard_centers(
        segments,
        (
            {
                "video_id": "scene::01",
                "frame_indices": [1, 3, 5],
                "status": "accept",
            },
            {
                "video_id": "scene::01",
                "frame_indices": [7, 9, 11],
                "status": "accept",
            },
        ),
    )
    assert retained == segments

    clipped = finalize_module._retain_segments_with_hard_centers(
        (finalize_module.FrameInterval("scene::01", 1, 3),),
        (
            {
                "video_id": "scene::01",
                "frame_indices": [1, 3, 5],
                "status": "accept",
            },
        ),
    )
    assert clipped == ()


def test_finalize_rejects_tampered_immutable_index_fields(tmp_path):
    _, _, config_path, main = _completed_case(tmp_path)
    records = list(read_jsonl(main.manifest_path))
    records[0]["img0"] = {**records[0]["img0"], "path": "/tmp/forged.png"}
    write_jsonl_part(main.manifest_path, records)

    with pytest.raises(RuntimeError, match="changed frozen index field 'img0'"):
        finalize_run(config_path)


def test_finalize_publication_rolls_back_all_outputs_if_current_commit_fails(
    tmp_path, monkeypatch
):
    root, _, config_path, _ = _completed_case(tmp_path)
    first = finalize_run(config_path)
    manifest_before = first.manifest_path.read_bytes()
    current_path = root / ".vfi_hard_miner_current.json"
    current_before = current_path.read_bytes()
    hard_before = {
        path.relative_to(root / "extremely_hard_case").as_posix(): path.read_bytes()
        for path in (root / "extremely_hard_case").rglob("*.png")
    }
    original_atomic_json = finalize_module._atomic_json

    def fail_current(path, payload):
        if Path(path).name == ".vfi_hard_miner_current.json":
            raise RuntimeError("injected CURRENT failure")
        return original_atomic_json(path, payload)

    monkeypatch.setattr(finalize_module, "_atomic_json", fail_current)
    with pytest.raises(RuntimeError, match="injected CURRENT failure"):
        finalize_run(config_path)

    assert first.manifest_path.read_bytes() == manifest_before
    assert current_path.read_bytes() == current_before
    assert {
        path.relative_to(root / "extremely_hard_case").as_posix(): path.read_bytes()
        for path in (root / "extremely_hard_case").rglob("*.png")
    } == hard_before
    assert not list(root.glob(".*.vfi-backup"))


def test_finalize_requeues_a_done_diagnostic_when_its_artifact_was_lost(tmp_path):
    _, config, config_path, _ = _completed_case(tmp_path)
    finalize_run(config_path)
    result_path = Path(config.runtime.run_dir) / "diagnostic_results.jsonl"
    first_results = list(read_jsonl(result_path))
    lost = Path(first_results[0]["artifact_path"])
    lost.unlink()

    finalize_run(config_path)
    recovered = {
        record["sample_id"]: record for record in read_jsonl(result_path)
    }[first_results[0]["sample_id"]]
    assert recovered["attempt"] == first_results[0]["attempt"] + 1
    assert Path(recovered["artifact_path"]).is_file()
