"""Idempotent materialization of selected original frames."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Literal


LinkMode = Literal["hardlink_then_copy", "copy"]


class OutputCollisionError(RuntimeError):
    """Raised when two source files would overwrite one output name."""


def _files_identical(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    chunk_size = 1024 * 1024
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(chunk_size)
            right_chunk = right_handle.read(chunk_size)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def link_or_copy(source: str | Path, destination: str | Path, *, mode: LinkMode) -> str:
    """Atomically hardlink or copy one original file.

    The operation is idempotent.  An existing byte-identical destination is
    retained; a different file at the same path is never overwritten.
    """

    source_path = Path(source).resolve()
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        if destination_path.is_file() and _files_identical(source_path, destination_path):
            return "existing"
        raise OutputCollisionError(
            f"destination already exists with different content: {destination_path}"
        )
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination_path.name}.", dir=destination_path.parent, delete=True
    ) as handle:
        temporary = Path(handle.name)
    method = "copy"
    try:
        if mode == "hardlink_then_copy":
            try:
                os.link(source_path, temporary)
                method = "hardlink"
            except OSError as exc:
                if exc.errno not in {
                    errno.EXDEV,
                    errno.EPERM,
                    errno.EACCES,
                    errno.ENOTSUP,
                    errno.EINVAL,
                }:
                    raise
        elif mode != "copy":
            raise ValueError(f"unknown link mode: {mode}")
        if method == "copy":
            shutil.copy2(source_path, temporary)
        os.replace(temporary, destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return method


def relative_output_path(
    source: str | Path,
    *,
    source_root: str | Path,
    layout: Literal["preserve_relative", "flat"] = "preserve_relative",
) -> Path:
    source_path = Path(source).resolve()
    root = Path(source_root).resolve()
    try:
        relative = source_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source is outside source_root: {source_path}") from exc
    if layout == "preserve_relative":
        return relative
    if layout == "flat":
        return Path(source_path.name)
    raise ValueError(f"unknown output layout: {layout}")


def materialize_original_frames(
    sources: Iterable[str | Path],
    *,
    source_root: str | Path,
    output_root: str | Path,
    mode: LinkMode = "hardlink_then_copy",
    layout: Literal["preserve_relative", "flat"] = "preserve_relative",
) -> dict[str, int]:
    """Materialize selected originals and return per-method counts."""

    root = Path(output_root)
    counts = {"hardlink": 0, "copy": 0, "existing": 0}
    seen_destinations: dict[Path, Path] = {}
    for source in sorted((Path(item).resolve() for item in sources), key=lambda item: item.as_posix()):
        relative = relative_output_path(source, source_root=source_root, layout=layout)
        destination = root / relative
        previous = seen_destinations.get(relative)
        if previous is not None and previous != source:
            raise OutputCollisionError(
                f"multiple source frames map to {relative}: {previous} and {source}"
            )
        seen_destinations[relative] = source
        method = link_or_copy(source, destination, mode=mode)
        counts[method] += 1
    return counts


def materialize_mapped_frames(
    mappings: Iterable[tuple[str | Path, str | Path]],
    *,
    output_root: str | Path,
    mode: LinkMode = "hardlink_then_copy",
) -> dict[str, int]:
    """Materialize explicit safe relative paths, preserving segment boundaries."""

    root = Path(output_root).resolve()
    counts = {"hardlink": 0, "copy": 0, "existing": 0}
    seen_destinations: dict[Path, Path] = {}
    ordered = sorted(
        ((Path(source).resolve(), Path(relative)) for source, relative in mappings),
        key=lambda item: (item[1].as_posix(), item[0].as_posix()),
    )
    for source, relative in ordered:
        if relative.is_absolute() or relative.drive or not relative.parts or ".." in relative.parts:
            raise ValueError(f"mapped output path must be a safe relative path: {relative}")
        destination = (root / relative).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"mapped output path escapes output_root: {relative}") from exc
        previous = seen_destinations.get(relative)
        if previous is not None and previous != source:
            raise OutputCollisionError(
                f"multiple source frames map to {relative}: {previous} and {source}"
            )
        seen_destinations[relative] = source
        method = link_or_copy(source, destination, mode=mode)
        counts[method] += 1
    return counts
