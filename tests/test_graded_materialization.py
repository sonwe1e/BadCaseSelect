from __future__ import annotations

from pathlib import Path

import pytest

from vfi_hard_miner.config import (
    AppConfig,
    CGVQMConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
)
from vfi_hard_miner.graded_materialization import (
    GradedMaterializer,
    plan_graded_mappings,
)
from vfi_hard_miner.outputs import OutputCollisionError


def _frame(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    return {"path": str(path)}


def _record(video: str, grade: str, frames) -> dict[str, object]:
    return {
        "sample_id": f"{video}-{grade}",
        "video_id": video,
        "grade": grade,
        "img0": frames[0],
        "gt": frames[1],
        "img1": frames[2],
    }


def _config(root: Path, run_dir: Path) -> AppConfig:
    return AppConfig(
        data=DataConfig(root=str(root)),
        model=ModelConfig(factory="mock"),
        cgvqm=CGVQMConfig(enabled=True),
        output=OutputConfig(
            layout="graded_flat",
            link_mode="copy",
            materialize_strategy="per_video",
        ),
    )


def test_123_to_a_and_345_to_b_keep_shared_frame_in_both_grades(tmp_path):
    frames = [
        _frame(tmp_path / f"01{index:05d}.png", str(index).encode("ascii"))
        for index in range(1, 6)
    ]
    records = (
        _record("01", "A", frames[:3]),
        _record("01", "B", frames[2:]),
    )

    mappings, centers = plan_graded_mappings(records)

    assert centers == {"A": 1, "B": 1}
    relative = {path.as_posix() for _, path in mappings}
    assert relative == {
        "A/0100001.png",
        "A/0100002.png",
        "A/0100003.png",
        "B/0100003.png",
        "B/0100004.png",
        "B/0100005.png",
    }


def test_flat_same_name_from_different_sources_fails_before_overwrite(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    left = _frame(left_dir / "0100001.png", b"left")
    right = _frame(right_dir / "0100001.png", b"right")
    other = _frame(tmp_path / "0100002.png", b"other")

    with pytest.raises(OutputCollisionError, match="multiple source"):
        plan_graded_mappings(
            (
                _record("01", "A", (left, other, other)),
                _record("02", "A", (right, other, other)),
            )
        )


def test_materializer_copies_original_bytes_and_resumes_without_recopy(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    frames = [
        _frame(data_root / f"01{index:05d}.png", bytes([index]) * (index + 3))
        for index in range(1, 4)
    ]
    records = [_record("01", "A", frames)]
    config = _config(data_root, tmp_path / "run")
    materializer = GradedMaterializer(
        config,
        execution_id="execution",
        run_dir=tmp_path / "run",
    )

    materializer.materialize_video("01", records)
    first = materializer.summary()
    materializer.materialize_all(records)
    second = materializer.summary()

    assert first.copy_counts["copy"] == 3
    assert second.copy_counts == first.copy_counts
    assert not list((materializer.hard_staging / "A").glob("*/*"))
    for source, relative in plan_graded_mappings(records)[0]:
        assert (materializer.hard_staging / relative).read_bytes() == source.read_bytes()
