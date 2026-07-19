"""Atomic, deterministic JSONL part files and final k-way merging."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import heapq
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


Record = Mapping[str, Any]
SortKey = Callable[[Record], Any]
IdentityKey = Callable[[Record], str | None]

DEFAULT_SORT_FIELDS = (
    "game_id",
    "game",
    "scene_id",
    "scene",
    "video_id",
    "video",
    "start",
    "frame_start",
    "end",
    "frame_end",
    "sample_id",
    "segment_id",
    "task_id",
)


class ManifestError(ValueError):
    """Base class for malformed or conflicting manifest data."""


class ManifestConflictError(ManifestError):
    """Two different records claim the same stable identity."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def canonical_json(record: Record) -> str:
    """Serialize one record independent of dictionary insertion order."""

    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _sortable(value: Any) -> tuple[int, Any]:
    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, float)):
        return (2, value)
    return (3, str(value))


def default_record_sort_key(record: Record) -> tuple[Any, ...]:
    """A deterministic domain-oriented key with canonical JSON tie-breaking."""

    return tuple(_sortable(record.get(field)) for field in DEFAULT_SORT_FIELDS) + (
        canonical_json(record),
    )


def record_identity(record: Record) -> str | None:
    for field in ("sample_id", "segment_id", "task_id"):
        value = record.get(field)
        if value is not None:
            return f"{field}:{value}"
    return None


def _atomic_text_path(destination: Path) -> tuple[int, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    return descriptor, Path(temporary)


def _fsync_parent_directory(destination: Path) -> None:
    """Best-effort rename durability on POSIX; directory handles fail on Windows."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(destination.parent, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _commit_atomic(file_descriptor: int, temporary: Path, destination: Path) -> None:
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    os.replace(temporary, destination)
    _fsync_parent_directory(destination)


def write_jsonl_part(
    path: str | Path,
    records: Iterable[Record],
    *,
    sort_key: SortKey = default_record_sort_key,
) -> int:
    """Sort and atomically replace one worker-owned JSONL part."""

    destination = Path(path)
    ordered = sorted((dict(record) for record in records), key=sort_key)
    descriptor, temporary = _atomic_text_path(destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n", closefd=False) as stream:
            for record in ordered:
                stream.write(canonical_json(record))
                stream.write("\n")
            stream.flush()
        _commit_atomic(descriptor, temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return len(ordered)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(
                    f"invalid JSON in {source} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ManifestError(
                    f"manifest record in {source} at line {line_number} is not an object"
                )
            yield record


def _checked_sorted_records(path: Path, sort_key: SortKey) -> Iterator[dict[str, Any]]:
    previous: Any | None = None
    have_previous = False
    for line_number, record in enumerate(read_jsonl(path), start=1):
        key = sort_key(record)
        if have_previous and key < previous:
            raise ManifestError(
                f"JSONL part {path} is not sorted at data record {line_number}; "
                "write parts with write_jsonl_part before merging"
            )
        previous = key
        have_previous = True
        yield record


def merge_jsonl_parts(
    part_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    sort_key: SortKey = default_record_sort_key,
    identity_key: IdentityKey | None = record_identity,
) -> int:
    """Atomically k-way merge pre-sorted parts into one stable JSONL file.

    Exact duplicate identities are emitted once.  Different payloads with the
    same stable identity fail loudly instead of making output depend on worker
    completion order.
    """

    destination = Path(output_path)
    sources = sorted(
        (Path(path) for path in part_paths),
        key=lambda path: (path.as_posix().casefold(), path.as_posix()),
    )
    iterators = [_checked_sorted_records(path, sort_key) for path in sources]
    heap: list[tuple[Any, str, int, int, dict[str, Any]]] = []
    positions = [0] * len(iterators)
    for source_index, iterator in enumerate(iterators):
        try:
            record = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (sort_key(record), canonical_json(record), source_index, 0, record),
        )

    descriptor, temporary = _atomic_text_path(destination)
    emitted = 0
    seen: dict[str, str] = {}
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n", closefd=False) as stream:
            while heap:
                _, canonical, source_index, _, record = heapq.heappop(heap)
                identity = None if identity_key is None else identity_key(record)
                if identity is not None and identity in seen:
                    if seen[identity] != canonical:
                        raise ManifestConflictError(
                            f"conflicting records for stable identity {identity!r}"
                        )
                else:
                    if identity is not None:
                        seen[identity] = canonical
                    stream.write(canonical)
                    stream.write("\n")
                    emitted += 1

                iterator = iterators[source_index]
                try:
                    following = next(iterator)
                except StopIteration:
                    continue
                positions[source_index] += 1
                heapq.heappush(
                    heap,
                    (
                        sort_key(following),
                        canonical_json(following),
                        source_index,
                        positions[source_index],
                        following,
                    ),
                )
            stream.flush()
        _commit_atomic(descriptor, temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return emitted


# Concise aliases for pipeline/finalize callers.
write_part = write_jsonl_part
merge_parts = merge_jsonl_parts
