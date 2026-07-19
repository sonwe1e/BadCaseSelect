from __future__ import annotations

import json
from pathlib import Path

import pytest

from vfi_hard_miner.manifest import (
    ManifestConflictError,
    ManifestError,
    canonical_json,
    merge_jsonl_parts,
    read_jsonl,
    write_jsonl_part,
)


def test_part_write_is_sorted_canonical_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "parts" / "worker-0.jsonl"
    records = [
        {"sample_id": "b", "video_id": "v", "start": 3, "unicode": "路灯"},
        {"start": 1, "video_id": "v", "sample_id": "a"},
    ]

    assert write_jsonl_part(path, records) == 2

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["sample_id"] for line in lines] == ["a", "b"]
    assert lines[0] == canonical_json(records[1])
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_k_way_merge_is_stable_independent_of_part_argument_order(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    duplicate = {"sample_id": "s2", "video_id": "v", "start": 2}
    write_jsonl_part(first, [duplicate, {"sample_id": "s4", "video_id": "v", "start": 4}])
    write_jsonl_part(second, [{"sample_id": "s1", "video_id": "v", "start": 1}, duplicate])

    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    assert merge_jsonl_parts([first, second], one) == 3
    assert merge_jsonl_parts([second, first], two) == 3

    assert one.read_bytes() == two.read_bytes()
    assert [record["sample_id"] for record in read_jsonl(one)] == ["s1", "s2", "s4"]


def test_conflicting_identity_does_not_replace_existing_destination(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    output = tmp_path / "manifest.jsonl"
    output.write_text("old-content\n", encoding="utf-8")
    write_jsonl_part(first, [{"sample_id": "same", "score": 1}])
    write_jsonl_part(second, [{"sample_id": "same", "score": 2}])

    with pytest.raises(ManifestConflictError, match="same"):
        merge_jsonl_parts([first, second], output)

    assert output.read_text(encoding="utf-8") == "old-content\n"


def test_unsorted_external_part_is_rejected_without_partial_output(tmp_path: Path) -> None:
    part = tmp_path / "bad.jsonl"
    part.write_text(
        canonical_json({"sample_id": "b", "start": 2})
        + "\n"
        + canonical_json({"sample_id": "a", "start": 1})
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"

    with pytest.raises(ManifestError, match="not sorted"):
        merge_jsonl_parts([part], output)

    assert not output.exists()


def test_empty_part_set_atomically_creates_empty_manifest(tmp_path: Path) -> None:
    output = tmp_path / "manifest.jsonl"
    assert merge_jsonl_parts([], output) == 0
    assert output.read_bytes() == b""
