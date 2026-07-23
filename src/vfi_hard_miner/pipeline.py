"""Stage orchestration for deterministic offline hard-case mining."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any, Iterable, Iterator

from .config import AppConfig
from .indexing import FrameTriplet, build_index
from .materialization import (
    IncrementalMaterializer,
    MaterializationSummary,
    STAGING_DIRECTORY,
)
from .manifest import canonical_json, read_jsonl, write_jsonl_part
from .manifest import merge_jsonl_parts
from .runtime import get_spawn_context, spawn_device_workers
from .state import TaskRecord, TaskStore


@dataclass(frozen=True, slots=True)
class IndexSummary:
    run_hash: str
    triplets: int
    videos: int
    chunks: int
    inserted_tasks: int
    index_path: Path
    state_path: Path


@dataclass(frozen=True, slots=True)
class MainStageSummary:
    run_hash: str
    workers: int
    records: int
    counts: dict[str, int]
    manifest_path: Path
    materialization: MaterializationSummary | None = None


@dataclass(frozen=True, slots=True)
class TeacherStageSummary:
    run_hash: str
    workers: int
    candidates: int
    records: int
    counts: dict[str, int]
    manifest_path: Path


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolved_runtime_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def run_directory(config: AppConfig) -> Path:
    return _resolved_runtime_path(config.runtime.run_dir)


def execution_snapshot_path(config: AppConfig) -> Path:
    return run_directory(config) / "execution.snapshot.json"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_snapshot(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configured checkpoint is missing: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "sha256": _sha256_path(path),
    }


def _source_tree_snapshot() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    files = sorted(package_root.rglob("*.py"), key=lambda path: path.as_posix())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_path(path)))
    return {"root": str(package_root), "files": len(files), "sha256": digest.hexdigest()}


def _factory_source_snapshot(factory: str) -> dict[str, Any] | None:
    module_name = factory.partition(":")[0]
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    origin = Path(spec.origin).resolve()
    if not origin.is_file():
        return None
    return {"module": module_name, "path": str(origin), "sha256": _sha256_path(origin)}


def _base_index_digest(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted((dict(item) for item in records), key=_index_sort_key):
        record.pop("execution_id", None)
        digest.update(canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _execution_snapshot_payload(
    config: AppConfig,
    *,
    index_content_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "config_hash": config.run_hash(),
        "index_content_sha256": index_content_sha256,
        "current_checkpoint": _checkpoint_snapshot(config.model.checkpoint),
        "teacher_checkpoint": (
            None if config.teacher is None else _checkpoint_snapshot(config.teacher.checkpoint)
        ),
        "current_factory_source": _factory_source_snapshot(config.model.factory),
        "teacher_factory_source": (
            None if config.teacher is None else _factory_source_snapshot(config.teacher.factory)
        ),
        "miner_source": _source_tree_snapshot(),
    }
    payload["execution_id"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return payload


def load_execution_snapshot(config: AppConfig) -> dict[str, Any]:
    path = execution_snapshot_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"execution snapshot is missing; run index first: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("execution_id"):
        raise RuntimeError(f"execution snapshot is malformed: {path}")
    return payload


def execution_id(config: AppConfig) -> str:
    return str(load_execution_snapshot(config)["execution_id"])


def run_state_path(config: AppConfig, *, stage: str = "main") -> Path:
    configured = _resolved_runtime_path(config.runtime.state_db)
    suffix = configured.suffix or ".sqlite3"
    stem = configured.name[: -len(configured.suffix)] if configured.suffix else configured.name
    snapshot = execution_snapshot_path(config)
    identity = config.run_hash()
    if snapshot.is_file():
        identity = f"{identity}-{execution_id(config)}"
    return configured.with_name(f"{stem}-{identity}-{stage}{suffix}")


def _frame_payload(frame: Any) -> dict[str, Any]:
    return {
        "frame_index": int(frame.index),
        "path": str(frame.path.resolve()),
        "relative_path": frame.relative_path,
        "size": int(frame.size),
        "mtime_ns": int(frame.mtime_ns),
    }


def serialize_triplet(triplet: FrameTriplet, *, run_hash: str) -> dict[str, Any]:
    return {
        "run_hash": run_hash,
        "sample_id": triplet.sample_id,
        "video_id": triplet.video_id,
        "stride": int(triplet.stride),
        "frame_indices": list(triplet.frame_indices),
        "img0": _frame_payload(triplet.img0),
        "gt": _frame_payload(triplet.gt),
        "img1": _frame_payload(triplet.img1),
    }


def _index_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("video_id", "")),
        tuple(int(value) for value in record.get("frame_indices", ())),
        str(record.get("sample_id", "")),
    )


def _chunk_task_id(
    run_hash: str,
    execution: str,
    stage: str,
    video_id: str,
    records: list[dict[str, Any]],
) -> str:
    identity = {
        "run_hash": run_hash,
        "execution_id": execution,
        "stage": stage,
        "video_id": video_id,
        "sample_ids": [record["sample_id"] for record in records],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{run_hash}:{stage}:{digest}"


def _chunks(values: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _write_config_snapshot(config: AppConfig, destination: Path) -> None:
    canonical = config.canonical_json()
    if destination.exists():
        existing = json.dumps(
            json.loads(destination.read_text(encoding="utf-8")),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if existing != canonical:
            raise RuntimeError(
                f"run directory contains a different configuration: {destination}"
            )
        return
    formatted = json.dumps(json.loads(canonical), ensure_ascii=False, indent=2) + "\n"
    _atomic_text(destination, formatted)


def build_run_index(config: AppConfig) -> IndexSummary:
    """Scan once, write a stable index, and enqueue video-chunk tasks."""

    config.validate()
    run_hash = config.run_hash()
    run_dir = run_directory(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_config_snapshot(config, run_dir / "config.snapshot.json")
    triplets = build_index(
        config.data.root,
        stride=config.data.stride,
        frame_regex=config.data.frame_regex,
        frame_digits=config.data.frame_digits,
        extensions=config.data.extensions,
        exclude_dirs=tuple(
            dict.fromkeys(
                (
                    *config.data.excluded_dirs,
                    config.output.hard_case_dir,
                    config.output.visualization_dir,
                    STAGING_DIRECTORY,
                )
            )
        ),
        recursive=config.data.recursive,
    )
    base_records = [serialize_triplet(triplet, run_hash=run_hash) for triplet in triplets]
    snapshot_payload = _execution_snapshot_payload(
        config,
        index_content_sha256=_base_index_digest(base_records),
    )
    snapshot_path = execution_snapshot_path(config)
    if snapshot_path.exists():
        existing_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if existing_snapshot != snapshot_payload:
            raise RuntimeError(
                "data, checkpoint, model source, or miner source changed inside an existing run; "
                "use a new runtime.run_dir/state_db instead of reusing stale results"
            )
    execution = str(snapshot_payload["execution_id"])
    records = [{**record, "execution_id": execution} for record in base_records]
    index_path = run_dir / "index.jsonl"
    if index_path.exists():
        existing_records = list(read_jsonl(index_path))
        if [canonical_json(record) for record in sorted(existing_records, key=_index_sort_key)] != [
            canonical_json(record) for record in sorted(records, key=_index_sort_key)
        ]:
            raise RuntimeError(
                "current frame index differs from the frozen run index; use a new run directory"
            )
    else:
        write_jsonl_part(index_path, records, sort_key=_index_sort_key)
    if not snapshot_path.exists():
        _atomic_text(
            snapshot_path,
            json.dumps(snapshot_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["video_id"]].append(record)
    tasks: list[tuple[str, dict[str, Any]]] = []
    chunk_index = 0
    for video_id in sorted(grouped):
        for chunk in _chunks(grouped[video_id], config.runtime.chunk_triplets):
            payload = {
                "run_hash": run_hash,
                "execution_id": execution,
                "stage": "main",
                "video_id": video_id,
                "chunk_index": chunk_index,
                "triplets": chunk,
            }
            tasks.append(
                (_chunk_task_id(run_hash, execution, "main", video_id, chunk), payload)
            )
            chunk_index += 1
    state_path = run_state_path(config, stage="main")
    with TaskStore(state_path) as store:
        inserted = store.enqueue_many(tasks)
    return IndexSummary(
        run_hash=run_hash,
        triplets=len(records),
        videos=len(grouped),
        chunks=len(tasks),
        inserted_tasks=inserted,
        index_path=index_path,
        state_path=state_path,
    )


def load_index_records(config: AppConfig) -> tuple[dict[str, Any], ...]:
    path = run_directory(config) / "index.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"run index is missing; run the index stage first: {path}")
    records = tuple(read_jsonl(path))
    expected_hash = config.run_hash()
    snapshot = load_execution_snapshot(config)
    expected_execution = str(snapshot["execution_id"])
    for record in records:
        if record.get("run_hash") != expected_hash:
            raise RuntimeError(f"index belongs to another configuration: {path}")
        if record.get("execution_id") != expected_execution:
            raise RuntimeError(f"index belongs to another execution snapshot: {path}")
    current_snapshot = _execution_snapshot_payload(
        config,
        index_content_sha256=_base_index_digest(records),
    )
    if current_snapshot != snapshot:
        raise RuntimeError(
            "frozen execution inputs changed after indexing; create a new run before mining"
        )
    seen_paths: set[str] = set()
    for record in records:
        for role in ("img0", "gt", "img1"):
            frame = record[role]
            frame_path = str(frame["path"])
            if frame_path in seen_paths:
                continue
            seen_paths.add(frame_path)
            source = Path(frame_path)
            try:
                stat = source.stat()
            except OSError as exc:
                raise RuntimeError(f"indexed frame is no longer readable: {source}") from exc
            if int(stat.st_size) != int(frame["size"]) or int(stat.st_mtime_ns) != int(
                frame["mtime_ns"]
            ):
                raise RuntimeError(
                    f"indexed frame changed after snapshot: {source}; create a new run"
                )
    return records


def stage_counts(config: AppConfig, *, stage: str = "main") -> dict[str, int]:
    state_path = run_state_path(config, stage=stage)
    if not state_path.exists():
        return {"pending": 0, "running": 0, "done": 0, "failed": 0}
    with TaskStore(state_path) as store:
        return store.counts()


def _stage_part_is_valid(config: AppConfig, stage: str, task: TaskRecord) -> bool:
    if task.status != "done" or task.result_path is None:
        return False
    suffix = task.task_id.rsplit(":", 1)[-1]
    expected = (
        run_directory(config)
        / f"{stage}_parts"
        / f"{suffix}.attempt-{task.attempt}.jsonl"
    ).resolve()
    if Path(task.result_path).resolve() != expected or not expected.is_file():
        return False
    payload_key = "triplets" if stage == "main" else "records"
    source = task.payload.get(payload_key)
    if not isinstance(source, list):
        return False
    expected_ids = {str(record.get("sample_id", "")) for record in source}
    try:
        results = list(read_jsonl(expected))
    except Exception:
        return False
    actual_ids = [str(record.get("sample_id", "")) for record in results]
    return (
        len(actual_ids) == len(expected_ids)
        and set(actual_ids) == expected_ids
        and all(record.get("execution_id") == execution_id(config) for record in results)
    )


def _recover_stage_results(config: AppConfig, stage: str) -> None:
    state_path = run_state_path(config, stage=stage)
    if not state_path.is_file():
        raise FileNotFoundError(f"{stage} state database is missing: {state_path}")
    with TaskStore(state_path) as store:
        store.recover_expired()
        for task in store.list_tasks():
            if task.status == "running" and task.owner is not None:
                owner_parts = task.owner.split(":", 2)
                if len(owner_parts) >= 2 and owner_parts[0] == socket.gethostname():
                    try:
                        owner_pid = int(owner_parts[1])
                        os.kill(owner_pid, 0)
                    except ProcessLookupError:
                        store.requeue_orphaned_running(task.task_id, task.owner)
                    except (ValueError, PermissionError):
                        pass
            elif task.status == "failed":
                store.requeue_failed(task.task_id)
            elif task.status == "done" and not _stage_part_is_valid(config, stage, task):
                store.requeue_done(task.task_id)


def _cpu_worker_bootstrap(worker_index: int, config_path: str) -> None:
    from .worker import main_worker_entry

    import torch

    main_worker_entry(worker_index, torch.device("cpu"), config_path)


def _cpu_teacher_bootstrap(worker_index: int, config_path: str) -> None:
    from .worker import teacher_worker_entry

    import torch

    teacher_worker_entry(worker_index, torch.device("cpu"), config_path)


def _materialize_ready_videos(
    config: AppConfig,
    materializer: IncrementalMaterializer,
) -> None:
    completed = materializer.completed_video_ids()
    grouped: dict[str, list[TaskRecord]] = defaultdict(list)
    with TaskStore(run_state_path(config, stage="main")) as store:
        for task in store.list_tasks():
            video_id = str(task.payload.get("video_id", ""))
            if video_id:
                grouped[video_id].append(task)
    for video_id in sorted(grouped):
        if video_id in completed:
            continue
        tasks = grouped[video_id]
        if not tasks or any(task.status != "done" for task in tasks):
            continue
        if any(not _stage_part_is_valid(config, "main", task) for task in tasks):
            continue
        materializer.materialize_tasks(video_id, tasks)


def run_main_stage(config_path: str | Path) -> MainStageSummary:
    """Run independent device workers and merge their atomic result parts."""

    from .worker import main_worker_entry

    path = Path(config_path).resolve()
    config = __import__("vfi_hard_miner.config", fromlist=["load_config"]).load_config(path)
    index_records = load_index_records(config)
    _recover_stage_results(config, "main")
    materializer = (
        IncrementalMaterializer(
            config,
            execution_id=execution_id(config),
            run_dir=run_directory(config),
            index_records=index_records,
        )
        if config.output.materialize_strategy == "per_video"
        else None
    )
    if config.runtime.backend == "cpu":
        context = get_spawn_context()
        processes = [
            context.Process(
                target=_cpu_worker_bootstrap,
                args=(index, str(path)),
                name=f"vfi-main-cpu-{index}",
            )
            for index in range(config.runtime.workers)
        ]
        for process in processes:
            process.start()
    else:
        devices = [
            f"{config.runtime.backend}:{index}"
            for index in config.runtime.devices[: config.runtime.workers]
        ]
        processes = spawn_device_workers(
            main_worker_entry,
            devices,
            args=(str(path),),
            join=False,
        )
    materialization_error: Exception | None = None
    while any(process.is_alive() for process in processes):
        if materializer is not None and materialization_error is None:
            try:
                _materialize_ready_videos(config, materializer)
            except Exception as exc:  # workers keep producing resumable result parts
                materialization_error = exc
        for process in processes:
            process.join(timeout=0.1)
        time.sleep(0.1)
    if materializer is not None and materialization_error is None:
        try:
            _materialize_ready_videos(config, materializer)
        except Exception as exc:
            materialization_error = exc
    failures = [
        f"{process.name}: exit code {process.exitcode}"
        for process in processes
        if process.exitcode != 0
    ]
    if failures:
        label = "CPU workers" if config.runtime.backend == "cpu" else "device workers"
        raise RuntimeError(f"one or more {label} failed: " + "; ".join(failures))
    if materialization_error is not None:
        raise RuntimeError(
            f"incremental materialization failed: {materialization_error}"
        ) from materialization_error
    state_path = run_state_path(config, stage="main")
    with TaskStore(state_path) as store:
        counts = store.counts()
        if counts["failed"] or counts["pending"] or counts["running"]:
            raise RuntimeError(f"main stage did not finish cleanly: {counts}")
        completed = store.list_tasks(status="done")
        invalid_parts = [
            task.task_id for task in completed if not _stage_part_is_valid(config, "main", task)
        ]
        if invalid_parts:
            raise RuntimeError(f"main stage has invalid winning parts: {invalid_parts[:3]}")
        part_paths = [Path(str(task.result_path)) for task in completed]
    manifest_path = run_directory(config) / "main_results.jsonl"
    records = merge_jsonl_parts(part_paths, manifest_path)
    return MainStageSummary(
        run_hash=config.run_hash(),
        workers=config.runtime.workers,
        records=records,
        counts=counts,
        manifest_path=manifest_path,
        materialization=None if materializer is None else materializer.summary(),
    )


def prepare_teacher_stage(config: AppConfig) -> int:
    if config.teacher is None:
        raise RuntimeError("teacher stage requires config.teacher")
    main_path = run_directory(config) / "main_results.jsonl"
    if not main_path.is_file():
        raise FileNotFoundError(f"main results are missing: {main_path}")
    candidates = [
        record for record in read_jsonl(main_path) if record.get("status") in {"accept", "review"}
    ]
    execution = execution_id(config)
    if any(record.get("execution_id") != execution for record in candidates):
        raise RuntimeError("main candidate belongs to another execution snapshot")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[str(record["video_id"])].append(record)
    tasks: list[tuple[str, dict[str, Any]]] = []
    chunk_index = 0
    for video_id in sorted(grouped):
        for chunk in _chunks(grouped[video_id], config.runtime.chunk_triplets):
            payload = {
                "run_hash": config.run_hash(),
                "execution_id": execution,
                "stage": "teacher",
                "video_id": video_id,
                "chunk_index": chunk_index,
                "records": chunk,
            }
            tasks.append(
                (
                    _chunk_task_id(
                        config.run_hash(), execution, "teacher", video_id, chunk
                    ),
                    payload,
                )
            )
            chunk_index += 1
    with TaskStore(run_state_path(config, stage="teacher")) as store:
        store.enqueue_many(tasks)
    _recover_stage_results(config, "teacher")
    return len(candidates)


def _teacher_candidate_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(record["sample_id"])
        for record in records
        if record.get("status") in {"accept", "review"}
    }


def _overlay_teacher_results(
    main_path: str | Path,
    update_path: str | Path | None,
    output_path: str | Path,
    *,
    expected_run_hash: str,
    expected_execution_id: str | None = None,
) -> int:
    """Atomically publish a full main manifest with teacher candidates replaced.

    Teacher workers intentionally process only ``accept``/``review`` candidates.
    Their output is therefore a delta, not a final manifest.  Keeping all main
    records here preserves reject records and, critically, invalid/out-of-scope
    intervals that act as forced segment barriers during finalization.
    """

    main_records = list(read_jsonl(main_path))
    main_by_id: dict[str, dict[str, Any]] = {}
    for record in main_records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise RuntimeError("main result contains a record without sample_id")
        if record.get("run_hash") != expected_run_hash:
            raise RuntimeError("main result belongs to another run")
        if (
            expected_execution_id is not None
            and record.get("execution_id") != expected_execution_id
        ):
            raise RuntimeError("main result belongs to another execution snapshot")
        if sample_id in main_by_id:
            raise RuntimeError(f"duplicate sample_id in main results: {sample_id}")
        main_by_id[sample_id] = record

    expected_updates = _teacher_candidate_ids(main_records)
    updates: dict[str, dict[str, Any]] = {}
    if update_path is not None:
        for record in read_jsonl(update_path):
            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                raise RuntimeError("teacher update contains a record without sample_id")
            if record.get("run_hash") != expected_run_hash:
                raise RuntimeError("teacher update belongs to another run")
            if (
                expected_execution_id is not None
                and record.get("execution_id") != expected_execution_id
            ):
                raise RuntimeError("teacher update belongs to another execution snapshot")
            original = main_by_id.get(sample_id)
            if original is None:
                raise RuntimeError(f"teacher update is absent from main results: {sample_id}")
            if sample_id in updates:
                raise RuntimeError(f"duplicate sample_id in teacher updates: {sample_id}")
            for field in ("video_id", "frame_indices", "img0", "gt", "img1", "stride"):
                if record.get(field) != original.get(field):
                    raise RuntimeError(
                        f"teacher update changed immutable field {field!r} for {sample_id}"
                    )
            updates[sample_id] = record

    actual_updates = set(updates)
    if actual_updates != expected_updates:
        missing = sorted(expected_updates - actual_updates)
        unexpected = sorted(actual_updates - expected_updates)
        raise RuntimeError(
            "teacher update set does not match main candidates; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    complete = [updates.get(str(record["sample_id"]), record) for record in main_records]
    return write_jsonl_part(output_path, complete)


def run_teacher_stage(config_path: str | Path) -> TeacherStageSummary:
    from .config import load_config
    from .worker import teacher_worker_entry

    path = Path(config_path).resolve()
    config = load_config(path)
    load_index_records(config)
    candidates = prepare_teacher_stage(config)
    manifest_path = run_directory(config) / "teacher_results.jsonl"
    if candidates == 0:
        records = _overlay_teacher_results(
            run_directory(config) / "main_results.jsonl",
            None,
            manifest_path,
            expected_run_hash=config.run_hash(),
            expected_execution_id=execution_id(config),
        )
        return TeacherStageSummary(
            config.run_hash(), config.runtime.workers, 0, records,
            {"done": 0, "failed": 0, "pending": 0, "running": 0}, manifest_path
        )
    if config.runtime.backend == "cpu":
        context = get_spawn_context()
        processes = [
            context.Process(
                target=_cpu_teacher_bootstrap,
                args=(index, str(path)),
                name=f"vfi-teacher-cpu-{index}",
            )
            for index in range(config.runtime.workers)
        ]
        for process in processes:
            process.start()
        failures: list[str] = []
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failures.append(f"{process.name}: exit code {process.exitcode}")
        if failures:
            raise RuntimeError("one or more teacher CPU workers failed: " + "; ".join(failures))
    else:
        devices = [
            f"{config.runtime.backend}:{index}"
            for index in config.runtime.devices[: config.runtime.workers]
        ]
        spawn_device_workers(
            teacher_worker_entry,
            devices,
            args=(str(path),),
            join=True,
        )
    with TaskStore(run_state_path(config, stage="teacher")) as store:
        counts = store.counts()
        if counts["failed"] or counts["pending"] or counts["running"]:
            raise RuntimeError(f"teacher stage did not finish cleanly: {counts}")
        completed = store.list_tasks(status="done")
        invalid_parts = [
            task.task_id
            for task in completed
            if not _stage_part_is_valid(config, "teacher", task)
        ]
        if invalid_parts:
            raise RuntimeError(f"teacher stage has invalid winning parts: {invalid_parts[:3]}")
        parts = [Path(str(task.result_path)) for task in completed]
    update_path = run_directory(config) / "teacher_updates.jsonl"
    merge_jsonl_parts(parts, update_path)
    records = _overlay_teacher_results(
        run_directory(config) / "main_results.jsonl",
        update_path,
        manifest_path,
        expected_run_hash=config.run_hash(),
        expected_execution_id=execution_id(config),
    )
    return TeacherStageSummary(
        config.run_hash(), config.runtime.workers, candidates, records, counts, manifest_path
    )


__all__ = [
    "IndexSummary",
    "MainStageSummary",
    "TeacherStageSummary",
    "build_run_index",
    "execution_id",
    "execution_snapshot_path",
    "load_execution_snapshot",
    "load_index_records",
    "run_directory",
    "run_main_stage",
    "run_teacher_stage",
    "prepare_teacher_stage",
    "run_state_path",
    "serialize_triplet",
    "stage_counts",
]
