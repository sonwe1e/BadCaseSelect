"""Verification and packaging of fully offline runtime resources."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


class BundleVerificationError(RuntimeError):
    """Raised when the offline manifest and files disagree."""


@dataclass(frozen=True, slots=True)
class ResourceRecord:
    name: str
    kind: str
    path: str
    source: str
    version: str
    license: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class ArchiveFileRecord:
    path: str
    sha256: str
    size: int


ARCHIVE_ROOT = "BadCaseSelect"
ARCHIVE_INVENTORY_PATH = f"{ARCHIVE_ROOT}/BUNDLE_INVENTORY.json"
DEFAULT_INCLUDE_ROOTS = (
    "src",
    "scripts",
    "configs",
    "docs",
    "third_party",
    "ckpts",
    "examples",
    "pyproject.toml",
    "requirements.lock",
    "requirements-build.lock",
    "README.md",
)
DEFAULT_REQUIRED_ROOTS = (
    "src",
    "scripts",
    "configs",
    "third_party",
    "pyproject.toml",
    "requirements.lock",
    "requirements-build.lock",
    "README.md",
)
RESOURCE_ONLY_ROOTS = (
    "third_party",
    "requirements.lock",
    "requirements-build.lock",
)
_MANAGED_RESOURCE_ROOTS = ("third_party", "ckpts")
_BUNDLE_METADATA_FILES = (
    "third_party/README.md",
    "third_party/manifest.json",
)
_GENERATED_DIRECTORY_NAMES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".pytest_tmp",
        ".mypy_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        ".git",
        ".tox",
        ".nox",
        ".venv",
        "venv",
        "build",
        "dist",
    }
)


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_resource_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise BundleVerificationError(f"unsafe resource path: {relative}")
    root_resolved = root.resolve()
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise BundleVerificationError(f"resource escapes bundle root: {relative}") from exc
    return resolved


def load_resource_manifest(path: str | Path) -> tuple[dict[str, Any], list[ResourceRecord]]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise BundleVerificationError("third_party manifest schema_version must be 1")
    resources = payload.get("resources")
    if not isinstance(resources, list):
        raise BundleVerificationError("third_party manifest resources must be an array")
    required = {"name", "kind", "path", "source", "version", "license", "sha256", "size"}
    records: list[ResourceRecord] = []
    for index, item in enumerate(resources):
        if not isinstance(item, dict):
            raise BundleVerificationError(f"resource {index} must be an object")
        missing = sorted(required - set(item))
        unknown = sorted(set(item) - required)
        if missing or unknown:
            raise BundleVerificationError(
                f"resource {index} fields invalid; missing={missing}, unknown={unknown}"
            )
        string_fields = ("name", "kind", "path", "source", "version", "license", "sha256")
        if any(not isinstance(item[field], str) or not item[field] for field in string_fields):
            raise BundleVerificationError(
                f"resource {index} string fields must be non-empty strings"
            )
        if isinstance(item["size"], bool) or not isinstance(item["size"], int):
            raise BundleVerificationError(f"resource {index} size must be an integer")
        record = ResourceRecord(**item)
        if len(record.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in record.sha256):
            raise BundleVerificationError(f"resource {record.name} has invalid lowercase SHA-256")
        if record.size < 0:
            raise BundleVerificationError(f"resource {record.name} has invalid size")
        records.append(record)
    paths = [record.path for record in records]
    if len(paths) != len(set(paths)):
        raise BundleVerificationError("third_party manifest contains duplicate paths")
    return payload, records


def verify_resource_manifest(
    manifest_path: str | Path,
    *,
    bundle_root: str | Path | None = None,
    require_nonempty: bool = False,
) -> list[ResourceRecord]:
    manifest = Path(manifest_path)
    root = Path(bundle_root) if bundle_root is not None else manifest.parent.parent
    _, records = load_resource_manifest(manifest)
    if require_nonempty and not records:
        raise BundleVerificationError("production bundle resource manifest must not be empty")
    problems: list[str] = []
    for record in records:
        path = _safe_resource_path(root, record.path)
        if not path.is_file():
            problems.append(f"missing: {record.path}")
            continue
        actual_size = path.stat().st_size
        if actual_size != record.size:
            problems.append(f"size mismatch: {record.path} expected={record.size} actual={actual_size}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != record.sha256:
            problems.append(f"SHA-256 mismatch: {record.path}")
    if problems:
        raise BundleVerificationError("offline resources failed verification:\n- " + "\n- ".join(problems))
    return records


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_roots(root: Path, values: Iterable[str]) -> tuple[Path, ...]:
    return tuple(_safe_resource_path(root, value) for value in values)


def _covered_by(path: Path, include_paths: Sequence[Path]) -> bool:
    resolved = path.resolve()
    for include in include_paths:
        include_resolved = include.resolve()
        if resolved == include_resolved:
            return True
        if include_resolved.is_dir() and _is_within(resolved, include_resolved):
            return True
    return False


def _validate_required_roots(
    root: Path,
    include_paths: Sequence[Path],
    required_roots: Iterable[str],
) -> None:
    missing: list[str] = []
    uncovered: list[str] = []
    for relative in required_roots:
        required = _safe_resource_path(root, relative)
        if not required.exists():
            missing.append(relative)
        elif not _covered_by(required, include_paths):
            uncovered.append(relative)
    if missing or uncovered:
        raise BundleVerificationError(
            f"bundle roots incomplete; missing={sorted(missing)}, "
            f"not_included={sorted(uncovered)}"
        )


def _configured_checkpoint_paths(root: Path) -> tuple[Path, ...]:
    configs_root = root / "configs"
    checkpoints: set[Path] = set()
    if not configs_root.is_dir():
        return ()
    for config_path in sorted(configs_root.rglob("*.json"), key=lambda item: item.as_posix()):
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleVerificationError(f"invalid JSON config: {config_path}") from exc
        if not isinstance(payload, dict):
            raise BundleVerificationError(f"config must be a JSON object: {config_path}")
        for section_name in ("model", "teacher"):
            section = payload.get(section_name)
            if section is None:
                continue
            if not isinstance(section, dict):
                raise BundleVerificationError(
                    f"config {config_path} section {section_name} must be an object or null"
                )
            checkpoint_value = section.get("checkpoint")
            if checkpoint_value in (None, ""):
                continue
            if not isinstance(checkpoint_value, str):
                raise BundleVerificationError(
                    f"config {config_path} {section_name}.checkpoint must be a string or null"
                )
            checkpoint = Path(checkpoint_value).expanduser()
            if not checkpoint.is_absolute() and ".." in checkpoint.parts:
                raise BundleVerificationError(
                    f"config checkpoint contains parent traversal: {checkpoint_value}"
                )
            checkpoint = (
                checkpoint.resolve()
                if checkpoint.is_absolute()
                else (root / checkpoint).resolve()
            )
            if not _is_within(checkpoint, root):
                raise BundleVerificationError(
                    f"configured checkpoint is outside project root: {checkpoint_value}"
                )
            checkpoints.add(checkpoint)
    return tuple(sorted(checkpoints, key=lambda item: item.as_posix()))


def verify_bundle_project(
    project_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    include_roots: Iterable[str] = DEFAULT_INCLUDE_ROOTS,
    required_roots: Iterable[str] = DEFAULT_REQUIRED_ROOTS,
) -> list[ResourceRecord]:
    """Validate all inputs needed by a production offline archive."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise BundleVerificationError(f"project root is not a directory: {root}")
    include_values = tuple(include_roots)
    include_paths = _resolve_roots(root, include_values)
    _validate_required_roots(root, include_paths, required_roots)
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else root / "third_party" / "manifest.json"
    )
    if not _is_within(manifest, root) or not _covered_by(manifest, include_paths):
        raise BundleVerificationError("resource manifest must be inside an included project root")
    records = verify_resource_manifest(
        manifest,
        bundle_root=root,
        require_nonempty=True,
    )
    record_paths = {record.path: record for record in records}
    for record in records:
        resource = _safe_resource_path(root, record.path)
        if not _covered_by(resource, include_paths):
            raise BundleVerificationError(
                f"verified resource is not covered by include_roots: {record.path}"
            )
    for checkpoint in _configured_checkpoint_paths(root):
        if not checkpoint.is_file():
            raise BundleVerificationError(f"configured checkpoint is missing: {checkpoint}")
        relative = checkpoint.relative_to(root).as_posix()
        if relative not in record_paths:
            raise BundleVerificationError(
                f"configured checkpoint is not listed in the resource manifest: {relative}"
            )
        if not _covered_by(checkpoint, include_paths):
            raise BundleVerificationError(
                f"configured checkpoint is not covered by include_roots: {relative}"
            )
    return records


def verify_resource_bundle_project(
    project_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> list[ResourceRecord]:
    """Validate the transferable dependency pack before user checkpoints exist."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise BundleVerificationError(f"project root is not a directory: {root}")
    include_paths = _resolve_roots(root, RESOURCE_ONLY_ROOTS)
    _validate_required_roots(root, include_paths, RESOURCE_ONLY_ROOTS)
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else root / "third_party" / "manifest.json"
    )
    if not _is_within(manifest, root) or not _covered_by(manifest, include_paths):
        raise BundleVerificationError("resource manifest must be inside third_party")
    records = verify_resource_manifest(
        manifest,
        bundle_root=root,
        require_nonempty=True,
    )
    for record in records:
        resource = _safe_resource_path(root, record.path)
        if not _covered_by(resource, include_paths):
            raise BundleVerificationError(
                f"verified resource is not covered by resource-only roots: {record.path}"
            )
    return records


def record_resource(
    manifest_path: str | Path,
    resource_path: str | Path,
    *,
    bundle_root: str | Path,
    name: str,
    kind: str,
    source: str,
    version: str,
    license_name: str,
) -> ResourceRecord:
    manifest = Path(manifest_path)
    root = Path(bundle_root)
    path = Path(resource_path).resolve()
    try:
        relative = path.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise BundleVerificationError("resource must be inside bundle_root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    payload, records = load_resource_manifest(manifest)
    record = ResourceRecord(
        name=name,
        kind=kind,
        path=relative,
        source=source,
        version=version,
        license=license_name,
        sha256=sha256_file(path),
        size=path.stat().st_size,
    )
    by_path = {item.path: item for item in records}
    by_path[record.path] = record
    payload["resources"] = [
        {
            "name": item.name,
            "kind": item.kind,
            "path": item.path,
            "source": item.source,
            "version": item.version,
            "license": item.license,
            "sha256": item.sha256,
            "size": item.size,
        }
        for item in sorted(by_path.values(), key=lambda value: value.path)
    ]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=manifest.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, manifest)
    return record


def _archive_payload_roots(
    root: Path,
    include_roots: Iterable[str],
    resources: Sequence[ResourceRecord],
) -> tuple[str, ...]:
    """Return an explicit, audited payload set for archive collection.

    Project code/documentation roots may be collected recursively.  The two
    namespaces that can contain downloaded or proprietary data, ``third_party``
    and ``ckpts``, are different: only manifest records plus fixed bundle
    metadata are distributable.  This prevents stale wheels, extracted demo
    media, caches, and unrelated checkpoints from entering an archive merely
    because they happen to exist in the working tree.
    """

    include_values = tuple(include_roots)
    include_paths = _resolve_roots(root, include_values)
    managed_roots = tuple((root / value).resolve() for value in _MANAGED_RESOURCE_ROOTS)
    payload: set[str] = set()
    for relative, path in zip(include_values, include_paths):
        resolved = path.resolve()
        ancestors = [
            managed.relative_to(root).as_posix()
            for managed in managed_roots
            if resolved != managed and _is_within(managed, resolved)
        ]
        if ancestors:
            raise BundleVerificationError(
                f"include root {relative!r} is an ancestor of managed resource roots "
                f"{sorted(ancestors)}; list project roots explicitly"
            )
        if any(resolved == managed or _is_within(resolved, managed) for managed in managed_roots):
            continue
        payload.add(Path(relative).as_posix())

    for relative in _BUNDLE_METADATA_FILES:
        path = _safe_resource_path(root, relative)
        if path.is_file() and _covered_by(path, include_paths):
            payload.add(relative)
    for record in resources:
        path = _safe_resource_path(root, record.path)
        if _covered_by(path, include_paths):
            payload.add(record.path)
    return tuple(sorted(payload))


def _collect_payload_files(root: Path, include_roots: Iterable[str]) -> tuple[Path, ...]:
    files: dict[str, Path] = {}
    for relative in include_roots:
        declared_source = root / Path(relative)
        if declared_source.is_symlink():
            raise BundleVerificationError(
                f"symlinks are not allowed in an offline archive: {relative}"
            )
        source = _safe_resource_path(root, relative)
        if not source.exists():
            continue
        candidates = (source,) if source.is_file() else tuple(source.rglob("*"))
        for candidate in candidates:
            if candidate.is_symlink():
                raise BundleVerificationError(
                    f"symlinks are not allowed in an offline archive: "
                    f"{candidate.relative_to(root)}"
                )
            if candidate.is_dir():
                continue
            relative_candidate = candidate.relative_to(root)
            if (
                any(part in _GENERATED_DIRECTORY_NAMES for part in relative_candidate.parts)
                or any(part.endswith(".egg-info") for part in relative_candidate.parts)
                or any(part.endswith(".dist-info") for part in relative_candidate.parts)
                or candidate.suffix in {".pyc", ".pyo"}
            ):
                continue
            if not candidate.is_file():
                raise BundleVerificationError(
                    f"unsupported filesystem entry in bundle: {candidate.relative_to(root)}"
                )
            resolved = candidate.resolve()
            if not _is_within(resolved, root):
                raise BundleVerificationError(f"bundle file escapes project root: {candidate}")
            relative_path = resolved.relative_to(root).as_posix()
            files[relative_path] = resolved
    return tuple(files[key] for key in sorted(files))


def _archive_inventory(
    root: Path,
    files: Sequence[Path],
    resources: Sequence[ResourceRecord],
) -> tuple[ArchiveFileRecord, ...]:
    verified_resources = {record.path: record for record in resources}
    inventory: list[ArchiveFileRecord] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        resource = verified_resources.get(relative)
        size = path.stat().st_size
        if resource is not None and resource.size == size:
            digest = resource.sha256
        else:
            digest = sha256_file(path)
        inventory.append(
            ArchiveFileRecord(
                path=f"{ARCHIVE_ROOT}/{relative}",
                sha256=digest,
                size=size,
            )
        )
    return tuple(inventory)


def _inventory_bytes(records: Sequence[ArchiveFileRecord]) -> bytes:
    payload = {
        "schema_version": 1,
        "archive_root": ARCHIVE_ROOT,
        "inventory_excludes_self": True,
        "files": [
            {"path": record.path, "sha256": record.sha256, "size": record.size}
            for record in records
        ],
    }
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _safe_archive_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise BundleVerificationError(f"unsafe archive member path: {name}")
    if path.parts[0] != ARCHIVE_ROOT:
        raise BundleVerificationError(f"archive member is outside {ARCHIVE_ROOT}: {name}")
    return path


def verify_offline_archive(path: str | Path) -> tuple[ArchiveFileRecord, ...]:
    """Read back an archive and verify every regular payload member."""

    archive_path = Path(path)
    try:
        archive = tarfile.open(archive_path, "r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise BundleVerificationError(f"cannot open offline archive: {archive_path}") from exc
    with archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise BundleVerificationError("offline archive contains duplicate member paths")
        for member in members:
            _safe_archive_name(member.name)
            if not member.isfile():
                raise BundleVerificationError(
                    f"offline archive contains a non-regular member: {member.name}"
                )
        try:
            inventory_member = archive.getmember(ARCHIVE_INVENTORY_PATH)
            inventory_handle = archive.extractfile(inventory_member)
            if inventory_handle is None:
                raise BundleVerificationError("offline archive inventory is unreadable")
            inventory_payload = json.load(inventory_handle)
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BundleVerificationError("offline archive inventory is missing or invalid") from exc
        if (
            not isinstance(inventory_payload, dict)
            or inventory_payload.get("schema_version") != 1
            or inventory_payload.get("archive_root") != ARCHIVE_ROOT
            or inventory_payload.get("inventory_excludes_self") is not True
            or not isinstance(inventory_payload.get("files"), list)
        ):
            raise BundleVerificationError("offline archive inventory schema is invalid")

        records: list[ArchiveFileRecord] = []
        for index, item in enumerate(inventory_payload["files"]):
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
                raise BundleVerificationError(f"archive inventory record {index} is invalid")
            if (
                not isinstance(item["path"], str)
                or not isinstance(item["sha256"], str)
                or len(item["sha256"]) != 64
                or any(ch not in "0123456789abcdef" for ch in item["sha256"])
                or isinstance(item["size"], bool)
                or not isinstance(item["size"], int)
                or item["size"] < 0
            ):
                raise BundleVerificationError(f"archive inventory record {index} is invalid")
            _safe_archive_name(item["path"])
            if item["path"] == ARCHIVE_INVENTORY_PATH:
                raise BundleVerificationError("archive inventory must exclude its own bytes")
            records.append(ArchiveFileRecord(**item))
        expected = {record.path: record for record in records}
        if len(expected) != len(records):
            raise BundleVerificationError("archive inventory contains duplicate paths")
        actual_names = set(names) - {ARCHIVE_INVENTORY_PATH}
        if actual_names != set(expected):
            raise BundleVerificationError(
                "archive members do not match inventory; "
                f"missing={sorted(set(expected) - actual_names)}, "
                f"unexpected={sorted(actual_names - set(expected))}"
            )
        for member_name, record in expected.items():
            member = archive.getmember(member_name)
            if member.size != record.size:
                raise BundleVerificationError(f"archive size mismatch: {member_name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise BundleVerificationError(f"archive member is unreadable: {member_name}")
            digest = hashlib.sha256()
            while chunk := handle.read(4 * 1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != record.sha256:
                raise BundleVerificationError(f"archive SHA-256 mismatch: {member_name}")
    return tuple(sorted(records, key=lambda record: record.path))


def build_offline_archive(
    project_root: str | Path,
    output_path: str | Path,
    *,
    include_roots: Iterable[str] = DEFAULT_INCLUDE_ROOTS,
    required_roots: Iterable[str] = DEFAULT_REQUIRED_ROOTS,
    resources_only: bool = False,
) -> Path:
    root = Path(project_root).resolve()
    if resources_only:
        include_values = RESOURCE_ONLY_ROOTS
        resources = verify_resource_bundle_project(root)
    else:
        include_values = tuple(include_roots)
        required_values = tuple(required_roots)
        resources = verify_bundle_project(
            root,
            include_roots=include_values,
            required_roots=required_values,
        )
    destination = Path(output_path).resolve()
    if destination.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("offline archive must end in .tar.gz")
    include_paths = tuple(
        path for path in _resolve_roots(root, include_values) if path.exists()
    )
    if any(
        destination == include.resolve()
        or (include.is_dir() and _is_within(destination, include.resolve()))
        for include in include_paths
    ):
        raise BundleVerificationError(
            "offline archive destination must be outside every included root"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload_roots = _archive_payload_roots(root, include_values, resources)
    files = _collect_payload_files(root, payload_roots)
    collected = {path.relative_to(root).as_posix() for path in files}
    omitted_resources = sorted(record.path for record in resources if record.path not in collected)
    if omitted_resources:
        raise BundleVerificationError(
            "verified manifest resources were omitted by archive collection: "
            f"{omitted_resources}"
        )
    inventory = _archive_inventory(root, files, resources)
    inventory_bytes = _inventory_bytes(inventory)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tar.gz", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with tarfile.open(
            temporary,
            "w:gz",
            format=tarfile.PAX_FORMAT,
            dereference=True,
        ) as archive:
            for path, record in zip(files, inventory):
                archive.add(path, arcname=record.path, recursive=False)
            information = tarfile.TarInfo(ARCHIVE_INVENTORY_PATH)
            information.size = len(inventory_bytes)
            information.mode = 0o644
            information.mtime = 0
            archive.addfile(information, io.BytesIO(inventory_bytes))
        verify_offline_archive(temporary)
        # Windows rejects fsync() on a descriptor opened read-only.
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
