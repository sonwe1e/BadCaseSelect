from __future__ import annotations

import re
from pathlib import Path

import pytest

from vfi_hard_miner.indexing import (
    DuplicateFrameError,
    FrameNameParser,
    build_index,
    consecutive_runs,
    iter_triplets,
    scan_videos,
)


def _touch(root: Path, *relative_paths: str) -> None:
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_frame_digits_group_multiple_videos_and_build_only_complete_triplets(
    tmp_path: Path,
) -> None:
    _touch(
        tmp_path,
        "scene_a/0100001.png",
        "scene_a/0100002.PNG",
        "scene_a/0100003.jpg",
        "scene_a/0100005.png",  # gap: must not bridge to a triplet
        "scene_a/0200001.jpeg",
        "scene_a/0200002.png",
        "scene_a/0200003.png",
        "scene_b/0100001.png",
        "scene_b/0100002.png",
        "scene_b/0100003.png",
    )

    videos = scan_videos(tmp_path, frame_digits=5)

    assert [video.video_id for video in videos] == [
        "scene_a::01",
        "scene_a::02",
        "scene_b::01",
    ]
    assert [frame.index for frame in videos[0].frames] == [1, 2, 3, 5]
    assert [triplet.frame_indices for triplet in iter_triplets(videos[0])] == [(1, 2, 3)]


def test_named_regex_can_include_extension_and_uses_named_video_group(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "area/clip-alpha-001.png",
        "area/clip-alpha-002.JPG",
        "area/clip-alpha-003.jpeg",
        "area/not-a-frame.png",
    )
    regex = r"^clip-(?P<video>[a-z]+)-(?P<frame>\d{3})\.(?P<ext>png|jpg|jpeg)$"

    videos = scan_videos(tmp_path, frame_regex=regex)

    assert len(videos) == 1
    assert videos[0].video_id == "area::alpha"
    assert len(tuple(iter_triplets(videos[0]))) == 1


def test_regex_without_video_group_infers_text_before_frame() -> None:
    parser = FrameNameParser(frame_regex=re.compile(r"^take_(?P<frame>\d{4})$"))
    parsed = parser.parse("take_0017.png")
    assert parsed is not None
    assert parsed.frame_index == 17
    assert parsed.video_key == "take_"


def test_output_and_hidden_directories_are_pruned(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "scene/0100001.png",
        "scene/0100002.png",
        "scene/0100003.png",
        "extremely_hard_case/9900001.png",
        "extremely_hard_case/9900002.png",
        "extremely_hard_case/9900003.png",
        "custom_output/8800001.png",
        ".cache/7700001.png",
    )

    videos = scan_videos(tmp_path, frame_digits=5, exclude_dirs=("custom_output",))

    assert [video.video_id for video in videos] == ["scene::01"]


def test_duplicate_numeric_frame_in_one_video_is_rejected(tmp_path: Path) -> None:
    _touch(tmp_path, "scene/0100001.png", "scene/0100001.jpg")
    with pytest.raises(DuplicateFrameError, match="duplicate frame index 1"):
        scan_videos(tmp_path, frame_digits=5)


def test_sample_ids_are_sha256_and_do_not_depend_on_absolute_root(tmp_path: Path) -> None:
    first_root = tmp_path / "one"
    second_root = tmp_path / "two"
    relative = ("scene/0100001.png", "scene/0100002.png", "scene/0100003.png")
    _touch(first_root, *relative)
    _touch(second_root, *relative)

    first = build_index(first_root, frame_digits=5)[0]
    second = build_index(second_root, frame_digits=5)[0]

    assert first.sample_id == second.sample_id
    assert len(first.sample_id) == 64
    int(first.sample_id, 16)


def test_consecutive_runs_and_non_unit_stride(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "scene/0100001.png",
        "scene/0100003.png",
        "scene/0100005.png",
        "scene/0100009.png",
    )
    video = scan_videos(tmp_path, frame_digits=5)[0]

    assert [[frame.index for frame in run] for run in consecutive_runs(video.frames, step=2)] == [
        [1, 3, 5],
        [9],
    ]
    assert [triplet.frame_indices for triplet in iter_triplets(video, stride=2)] == [
        (1, 3, 5)
    ]


def test_parser_configuration_is_unambiguous() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        FrameNameParser(frame_regex=r"(?P<frame>\d+)", frame_digits=5)
    with pytest.raises(ValueError, match="named 'frame'"):
        FrameNameParser(frame_regex=r"(\d+)")
    with pytest.raises(ValueError, match="positive integer"):
        FrameNameParser(frame_digits=2.5)  # type: ignore[arg-type]


def test_non_recursive_scan_only_uses_root_files(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "0100001.png",
        "0100002.png",
        "0100003.png",
        "nested/0200001.png",
        "nested/0200002.png",
        "nested/0200003.png",
    )
    videos = scan_videos(tmp_path, frame_digits=5, recursive=False)
    assert [video.video_id for video in videos] == ["::01"]
