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
from vfi_hard_miner.manifest import write_jsonl_part
from vfi_hard_miner.pipeline import (
    _overlay_teacher_results,
    build_run_index,
    load_index_records,
    run_main_stage,
    run_teacher_stage,
)
from vfi_hard_miner.segments import ClassifiedInterval, merge_classified_intervals


def test_teacher_stage_adds_recoverability_evidence(tmp_path):
    root = tmp_path / "game"
    root.mkdir()
    for index, value in enumerate((0, 128, 255), start=1):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = value
        Image.fromarray(image).save(root / f"01{index:05d}.png")
    common = dict(
        factory="vfi_hard_miner.mock_model:create_model",
        input_height=64,
        input_width=64,
        batch_size=2,
    )
    config = AppConfig(
        data=DataConfig(root=str(root)),
        model=ModelConfig(
            **common,
            factory_kwargs={"output_scale": 2, "endpoint_copy_box": [0.25, 0.25, 0.75, 0.75]},
        ),
        teacher=ModelConfig(**common, factory_kwargs={"output_scale": 2}),
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
    build_run_index(config)
    run_main_stage(config_path)
    summary = run_teacher_stage(config_path)
    assert summary.candidates == 1
    records = list(read_jsonl(summary.manifest_path))
    assert records[0]["teacher"]["local_error"] < records[0]["p_wrong"]
    assert records[0]["teacher"]["solvability"]["teacher_gain"] > 0.5


def _result(sample_id, status, indices, *, run_hash="run"):
    frames = {
        role: {"frame_index": index, "path": f"/{role}-{index}.png"}
        for role, index in zip(("img0", "gt", "img1"), indices)
    }
    return {
        "run_hash": run_hash,
        "sample_id": sample_id,
        "video_id": "scene::01",
        "stride": 1,
        "frame_indices": list(indices),
        **frames,
        "status": status,
        "reasons": [],
    }


def test_teacher_overlay_preserves_invalid_barrier_between_hard_samples(tmp_path):
    left = _result("left", "accept", (1, 2, 3))
    barrier = _result("barrier", "invalid", (3, 4, 5))
    right = _result("right", "accept", (5, 6, 7))
    left_update = {**left, "teacher": {"local_error": 0.1}}
    right_update = {**right, "teacher": {"local_error": 0.1}}
    main_path = tmp_path / "main.jsonl"
    update_path = tmp_path / "updates.jsonl"
    output_path = tmp_path / "teacher.jsonl"
    write_jsonl_part(main_path, [left, barrier, right])
    write_jsonl_part(update_path, [right_update, left_update])

    assert _overlay_teacher_results(
        main_path, update_path, output_path, expected_run_hash="run"
    ) == 3

    records = list(read_jsonl(output_path))
    assert {record["sample_id"] for record in records} == {"left", "barrier", "right"}
    assert next(record for record in records if record["sample_id"] == "barrier")[
        "status"
    ] == "invalid"
    segments = merge_classified_intervals(
        ClassifiedInterval(
            str(record["video_id"]),
            int(record["frame_indices"][0]),
            int(record["frame_indices"][-1]),
            str(record["status"]),
            str(record["sample_id"]),
        )
        for record in records
    )
    assert [(segment.start, segment.end) for segment in segments] == [(1, 2), (6, 7)]


def test_teacher_stage_with_zero_candidates_copies_full_main_manifest(tmp_path):
    root = tmp_path / "game"
    root.mkdir()
    for index in range(1, 5):
        Image.new("RGB", (8, 8), (index, index, index)).save(
            root / f"01{index:05d}.png"
        )
    config = AppConfig(
        data=DataConfig(root=str(root)),
        model=ModelConfig(factory="vfi_hard_miner.mock_model:create_model"),
        teacher=ModelConfig(factory="vfi_hard_miner.mock_model:create_model"),
        runtime=RuntimeConfig(
            backend="cpu",
            devices=(0,),
            workers=1,
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
        ),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(config.canonical_json(), encoding="utf-8")
    build_run_index(config)
    main_path = tmp_path / "run" / "main_results.jsonl"
    frozen = list(load_index_records(config))
    write_jsonl_part(
        main_path,
        [
            {**frozen[0], "status": "invalid", "reasons": []},
            {**frozen[1], "status": "reject", "reasons": []},
        ],
    )

    summary = run_teacher_stage(config_path)

    assert summary.candidates == 0
    assert summary.records == 2
    assert sorted(record["status"] for record in read_jsonl(summary.manifest_path)) == [
        "invalid", "reject"
    ]
