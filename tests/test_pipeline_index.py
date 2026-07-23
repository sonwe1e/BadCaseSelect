from __future__ import annotations

from PIL import Image
import pytest

from vfi_hard_miner.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    RuntimeConfig,
)
from vfi_hard_miner.pipeline import build_run_index, load_index_records, stage_counts


def _write_frame(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), (value, value, value)).save(path)


def test_build_run_index_is_deterministic_and_idempotent(tmp_path):
    frames = tmp_path / "game" / "scene"
    for index in range(1, 7):
        _write_frame(frames / f"01{index:05d}.png", index)
    config = AppConfig(
        data=DataConfig(root=str(tmp_path / "game")),
        model=ModelConfig(factory="examples.mock_model:create_model"),
        runtime=RuntimeConfig(
            backend="cpu",
            devices=(0,),
            workers=1,
            chunk_triplets=2,
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
        ),
    )
    first = build_run_index(config)
    second = build_run_index(config)
    assert first.triplets == 4
    assert first.videos == 1
    assert first.chunks == 2
    assert first.inserted_tasks == 2
    assert second.inserted_tasks == 0
    records = load_index_records(config)
    assert [record["frame_indices"] for record in records] == [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
    ]
    assert stage_counts(config) == {"done": 0, "failed": 0, "pending": 2, "running": 0}


def test_existing_run_rejects_changed_frame_snapshot(tmp_path):
    frames = tmp_path / "game"
    for index in range(1, 4):
        _write_frame(frames / f"01{index:05d}.png", index)
    config = AppConfig(
        data=DataConfig(root=str(frames)),
        model=ModelConfig(factory="examples.mock_model:create_model"),
        runtime=RuntimeConfig(
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
        ),
    )
    build_run_index(config)
    Image.new("RGB", (17, 13), (200, 200, 200)).save(frames / "0100002.png")
    with pytest.raises(RuntimeError, match="changed inside an existing run"):
        build_run_index(config)


def test_existing_run_rejects_checkpoint_content_change(tmp_path):
    frames = tmp_path / "game"
    for index in range(1, 4):
        _write_frame(frames / f"01{index:05d}.png", index)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"first checkpoint")
    config = AppConfig(
        data=DataConfig(root=str(frames)),
        model=ModelConfig(
            factory="examples.mock_model:create_model", checkpoint=str(checkpoint)
        ),
        runtime=RuntimeConfig(
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
        ),
    )
    build_run_index(config)
    checkpoint.write_bytes(b"second checkpoint with another size")
    with pytest.raises(RuntimeError, match="changed inside an existing run"):
        build_run_index(config)


def test_custom_generated_output_directories_are_always_excluded(tmp_path):
    root = tmp_path / "game"
    for directory in (root / "scene", root / "mined_frames", root / "mined_debug"):
        for index in range(1, 4):
            _write_frame(directory / f"01{index:05d}.png", index)
    config = AppConfig(
        data=DataConfig(root=str(root)),
        model=ModelConfig(factory="examples.mock_model:create_model"),
        runtime=RuntimeConfig(
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
        ),
        output=OutputConfig(
            hard_case_dir="mined_frames", visualization_dir="mined_debug"
        ),
    )
    summary = build_run_index(config)
    assert summary.triplets == 1
    assert summary.videos == 1


def test_incremental_staging_directory_is_always_excluded(tmp_path):
    root = tmp_path / "game"
    for directory in (
        root / "scene",
        root / ".vfi_hard_miner_staging" / "execution" / "hard_case" / "01",
    ):
        for index in range(1, 4):
            _write_frame(directory / f"01{index:05d}.png", index)
    config = AppConfig(
        data=DataConfig(root=str(root)),
        model=ModelConfig(factory="examples.mock_model:create_model"),
        runtime=RuntimeConfig(
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
        ),
    )

    summary = build_run_index(config)

    assert summary.triplets == 1
    assert summary.videos == 1
