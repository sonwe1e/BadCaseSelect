"""Deterministic discovery of numbered video frames and interpolation triplets.

The indexer deliberately does not depend on the package configuration or schema
modules.  This keeps directory discovery usable by small probe/finalize tools and
allows the exact same IDs to be generated on Windows and Linux.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterable, Iterator, Pattern, Sequence


IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})
DEFAULT_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {"extremely_hard_case", "extremely_hard_case_visualization"}
)


class IndexingError(ValueError):
    """Base class for deterministic-index construction errors."""


class DuplicateFrameError(IndexingError):
    """Raised when one logical video contains the same frame number twice."""


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ParsedFrameName:
    frame_index: int
    video_key: str


@dataclass(frozen=True, slots=True)
class FrameRecord:
    index: int
    path: Path
    relative_path: str
    size: int
    mtime_ns: int

    @property
    def frame_index(self) -> int:
        """Schema-friendly alias for callers using ``FrameRef`` terminology."""

        return self.index


@dataclass(frozen=True, slots=True)
class VideoSequence:
    """All numbered frames belonging to one source video."""

    video_id: str
    directory: Path
    video_key: str
    frames: tuple[FrameRecord, ...]


@dataclass(frozen=True, slots=True)
class FrameTriplet:
    """A midpoint-interpolation sample ``img0, gt, img1``."""

    sample_id: str
    video_id: str
    img0: FrameRecord
    gt: FrameRecord
    img1: FrameRecord
    stride: int

    @property
    def frame_indices(self) -> tuple[int, int, int]:
        return (self.img0.index, self.gt.index, self.img1.index)

    @property
    def video_key(self) -> str:
        return self.video_id

    @property
    def frame_interval(self) -> tuple[int, int]:
        return (self.img0.index, self.img1.index)


class FrameNameParser:
    """Parse either a named regular expression or a fixed trailing frame field.

    A regular expression must expose a named ``frame`` group.  A named ``video``
    (or ``prefix``) group is optional; without one, the text preceding ``frame``
    is used as the video key.  The expression is tried against the full filename
    first and then its stem, so both extension-aware and stem-only rules work.

    With ``frame_digits=N``, the final N digits are the frame number and the
    preceding stem is the video key.  For example, with five digits,
    ``0100001.png`` is video ``01`` frame ``1``.
    """

    def __init__(
        self,
        *,
        frame_regex: str | Pattern[str] | None = None,
        frame_digits: int | None = None,
    ) -> None:
        if (frame_regex is None) == (frame_digits is None):
            raise ValueError("configure exactly one of frame_regex or frame_digits")
        if frame_digits is not None:
            _require_positive_int(frame_digits, "frame_digits")
            self._pattern = re.compile(
                rf"^(?P<video>.*?)(?P<frame>\d{{{frame_digits}}})$"
            )
            self._extension_aware = False
        else:
            self._pattern = (
                frame_regex
                if isinstance(frame_regex, re.Pattern)
                else re.compile(frame_regex, re.IGNORECASE)  # type: ignore[arg-type]
            )
            if "frame" not in self._pattern.groupindex:
                raise ValueError("frame_regex must contain a named 'frame' group")
            self._extension_aware = True

    def parse(self, path: str | Path) -> ParsedFrameName | None:
        file_path = Path(path)
        candidates = (file_path.name, file_path.stem) if self._extension_aware else (file_path.stem,)
        match: re.Match[str] | None = None
        for candidate in candidates:
            match = self._pattern.fullmatch(candidate)
            if match is not None:
                break
        if match is None:
            return None

        raw_frame = match.group("frame")
        try:
            frame_index = int(raw_frame)
        except (TypeError, ValueError) as exc:
            raise IndexingError(f"non-integer frame group in {file_path.name!r}") from exc
        if frame_index < 0:
            raise IndexingError(f"negative frame number in {file_path.name!r}")

        groups = match.groupdict()
        video_key = groups.get("video") or groups.get("prefix")
        if video_key is None:
            video_key = match.string[: match.start("frame")]
        return ParsedFrameName(frame_index=frame_index, video_key=video_key)


def _canonical_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _normalise_extensions(extensions: Iterable[str]) -> frozenset[str]:
    normalised = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }
    if not normalised:
        raise ValueError("at least one image extension is required")
    return frozenset(normalised)


def _excluded_path_sets(
    root: Path, exclude_dirs: Iterable[str | Path]
) -> tuple[set[str], tuple[Path, ...]]:
    names = set(DEFAULT_EXCLUDED_DIRECTORY_NAMES)
    absolute_paths: list[Path] = []
    for value in exclude_dirs:
        candidate = Path(value)
        if candidate.is_absolute() or len(candidate.parts) > 1:
            absolute_paths.append(
                (candidate if candidate.is_absolute() else root / candidate).resolve()
            )
        else:
            names.add(candidate.name)
    return names, tuple(absolute_paths)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def scan_videos(
    root: str | Path,
    *,
    frame_regex: str | Pattern[str] | None = None,
    frame_digits: int | None = None,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
    exclude_dirs: Iterable[str | Path] = (),
    skip_hidden_dirs: bool = True,
    recursive: bool = True,
) -> tuple[VideoSequence, ...]:
    """Recursively scan *root* and group numbered images into logical videos.

    A video group is the pair ``(relative parent directory, parsed video key)``.
    Consequently two scenes may reuse the same video prefix without colliding.
    Output and hidden directories are pruned before their files are visited.
    """

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"frame root is not a directory: {root_path}")
    if frame_regex is None and frame_digits is None:
        frame_digits = 5
    parser = FrameNameParser(frame_regex=frame_regex, frame_digits=frame_digits)
    allowed_extensions = _normalise_extensions(extensions)
    excluded_names, excluded_paths = _excluded_path_sets(root_path, exclude_dirs)

    grouped: dict[tuple[str, str], list[FrameRecord]] = {}
    for current, directory_names, file_names in os.walk(root_path, topdown=True):
        current_path = Path(current)
        kept_directories: list[str] = []
        for directory_name in directory_names:
            child = (current_path / directory_name).resolve()
            if directory_name in excluded_names:
                continue
            if skip_hidden_dirs and directory_name.startswith("."):
                continue
            if any(_is_within(child, excluded) for excluded in excluded_paths):
                continue
            kept_directories.append(directory_name)
        directory_names[:] = sorted(kept_directories, key=lambda name: (name.casefold(), name))
        if not recursive:
            directory_names[:] = []

        relative_parent = current_path.relative_to(root_path).as_posix()
        for file_name in sorted(file_names, key=lambda name: (name.casefold(), name)):
            path = current_path / file_name
            if path.suffix.lower() not in allowed_extensions:
                continue
            parsed = parser.parse(path)
            if parsed is None:
                continue
            stat = path.stat()
            group_key = (relative_parent, parsed.video_key)
            grouped.setdefault(group_key, []).append(
                FrameRecord(
                    index=parsed.frame_index,
                    path=path,
                    relative_path=_canonical_relative(path, root_path),
                    size=int(stat.st_size),
                    mtime_ns=int(stat.st_mtime_ns),
                )
            )

    videos: list[VideoSequence] = []
    for (relative_parent, video_key), frames in grouped.items():
        frames.sort(key=lambda frame: (frame.index, frame.relative_path))
        for previous, current in zip(frames, frames[1:]):
            if previous.index == current.index:
                raise DuplicateFrameError(
                    "duplicate frame index "
                    f"{current.index} in {relative_parent!r}, video {video_key!r}: "
                    f"{previous.relative_path!r} and {current.relative_path!r}"
                )
        parent_key = relative_parent if relative_parent != "." else ""
        video_id = f"{parent_key}::{video_key}"
        videos.append(
            VideoSequence(
                video_id=video_id,
                directory=(root_path / relative_parent).resolve(),
                video_key=video_key,
                frames=tuple(frames),
            )
        )
    videos.sort(key=lambda video: (video.video_id.casefold(), video.video_id))
    return tuple(videos)


def consecutive_runs(
    frames: Sequence[FrameRecord], *, step: int = 1
) -> tuple[tuple[FrameRecord, ...], ...]:
    """Split sorted frames wherever the numeric step is not exactly *step*."""

    _require_positive_int(step, "step")
    if not frames:
        return ()
    ordered = sorted(frames, key=lambda frame: (frame.index, frame.relative_path))
    runs: list[list[FrameRecord]] = [[ordered[0]]]
    for frame in ordered[1:]:
        if frame.index - runs[-1][-1].index == step:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return tuple(tuple(run) for run in runs)


def stable_sample_id(
    video_id: str,
    frames: Sequence[FrameRecord],
    *,
    stride: int,
) -> str:
    """Return a cross-platform SHA-256 ID for one ordered sample."""

    if not frames:
        raise ValueError("a sample must contain at least one frame")
    _require_positive_int(stride, "stride")
    payload = {
        "version": 1,
        "video_id": video_id,
        "stride": stride,
        "frames": [
            {"index": frame.index, "path": frame.relative_path.replace("\\", "/")}
            for frame in frames
        ],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def iter_triplets(video: VideoSequence, *, stride: int = 1) -> Iterator[FrameTriplet]:
    """Yield all midpoint triplets whose three numeric frame indices exist."""

    _require_positive_int(stride, "stride")
    by_index = {frame.index: frame for frame in video.frames}
    if len(by_index) != len(video.frames):
        raise DuplicateFrameError(f"duplicate frame number in video {video.video_id!r}")
    for first_index in sorted(by_index):
        indices = (first_index, first_index + stride, first_index + 2 * stride)
        if not all(index in by_index for index in indices):
            continue
        frames = tuple(by_index[index] for index in indices)
        yield FrameTriplet(
            sample_id=stable_sample_id(video.video_id, frames, stride=stride),
            video_id=video.video_id,
            img0=frames[0],
            gt=frames[1],
            img1=frames[2],
            stride=stride,
        )


def build_index(
    root: str | Path,
    *,
    stride: int = 1,
    frame_regex: str | Pattern[str] | None = None,
    frame_digits: int | None = None,
    extensions: Iterable[str] = IMAGE_EXTENSIONS,
    exclude_dirs: Iterable[str | Path] = (),
    recursive: bool = True,
) -> tuple[FrameTriplet, ...]:
    """Convenience wrapper returning all triplets in deterministic order."""

    videos = scan_videos(
        root,
        frame_regex=frame_regex,
        frame_digits=frame_digits,
        extensions=extensions,
        exclude_dirs=exclude_dirs,
        recursive=recursive,
    )
    return tuple(triplet for video in videos for triplet in iter_triplets(video, stride=stride))


# A descriptive alias used by callers that think in terms of frame roots rather
# than logical videos.
scan_frames = scan_videos
