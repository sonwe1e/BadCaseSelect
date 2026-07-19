from __future__ import annotations

import os

import pytest

from vfi_hard_miner.outputs import (
    OutputCollisionError,
    link_or_copy,
    materialize_mapped_frames,
    materialize_original_frames,
)


def test_link_or_copy_is_idempotent(tmp_path):
    source = tmp_path / "source" / "0100001.png"
    source.parent.mkdir()
    source.write_bytes(b"frame")
    destination = tmp_path / "output" / source.name
    method = link_or_copy(source, destination, mode="hardlink_then_copy")
    assert method in {"hardlink", "copy"}
    assert destination.read_bytes() == b"frame"
    assert link_or_copy(source, destination, mode="hardlink_then_copy") == "existing"
    if method == "hardlink":
        assert os.stat(source).st_ino == os.stat(destination).st_ino


def test_flat_layout_detects_collisions(tmp_path):
    left = tmp_path / "source" / "a" / "0100001.png"
    right = tmp_path / "source" / "b" / "0100001.png"
    left.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    with pytest.raises(OutputCollisionError, match="multiple source"):
        materialize_original_frames(
            (left, right),
            source_root=tmp_path / "source",
            output_root=tmp_path / "output",
            mode="copy",
            layout="flat",
        )


def test_preserve_relative_keeps_scene_boundaries(tmp_path):
    left = tmp_path / "source" / "scene-a" / "0100001.png"
    right = tmp_path / "source" / "scene-b" / "0100001.png"
    left.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    counts = materialize_original_frames(
        (left, right),
        source_root=tmp_path / "source",
        output_root=tmp_path / "output",
        mode="copy",
    )
    assert counts["copy"] == 2
    assert (tmp_path / "output" / "scene-a" / left.name).read_bytes() == b"left"
    assert (tmp_path / "output" / "scene-b" / right.name).read_bytes() == b"right"


def test_mapped_layout_keeps_duplicate_source_frame_in_separate_segments(tmp_path):
    source = tmp_path / "source" / "0100003.png"
    source.parent.mkdir()
    source.write_bytes(b"shared-frame")

    counts = materialize_mapped_frames(
        (
            (source, "segment-a/0100003.png"),
            (source, "segment-b/0100003.png"),
        ),
        output_root=tmp_path / "output",
        mode="copy",
    )

    assert counts["copy"] == 2
    assert (tmp_path / "output/segment-a/0100003.png").read_bytes() == b"shared-frame"
    assert (tmp_path / "output/segment-b/0100003.png").read_bytes() == b"shared-frame"

    with pytest.raises(ValueError, match="safe relative path"):
        materialize_mapped_frames(
            ((source, "../escape.png"),),
            output_root=tmp_path / "output",
            mode="copy",
        )
