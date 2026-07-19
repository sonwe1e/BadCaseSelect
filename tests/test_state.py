from __future__ import annotations

from pathlib import Path

import pytest

from vfi_hard_miner.state import (
    LeaseLostError,
    TaskConflictError,
    TaskStore,
)


def test_store_uses_wal_and_enqueue_is_idempotent(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "state.sqlite3")
    assert store.connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    assert store.enqueue("a", {"video": "v", "start": 0}, now=1.0)
    assert not store.enqueue("a", {"start": 0, "video": "v"}, now=2.0)
    with pytest.raises(TaskConflictError):
        store.enqueue("a", {"video": "other"})
    assert store.counts() == {"done": 0, "failed": 0, "pending": 1, "running": 0}


def test_claim_is_priority_ordered_and_completion_is_idempotent(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "state.sqlite3")
    store.enqueue("low", {"n": 1}, priority=0, now=1.0)
    store.enqueue("high", {"n": 2}, priority=10, now=2.0)

    task = store.claim("worker-0", lease_seconds=30, now=10.0)
    assert task is not None
    assert task.task_id == "high"
    assert task.status == "running"
    assert task.attempt == 1
    assert task.lease_expires == 40.0

    assert store.heartbeat("high", "worker-0", lease_seconds=50, now=20.0) == 70.0
    assert store.complete("high", "worker-0", result_path="parts/high.jsonl", now=21.0)
    assert not store.complete("high", "any-owner", result_path="parts/high.jsonl", now=22.0)
    with pytest.raises(TaskConflictError):
        store.complete("high", "worker-0", result_path="parts/other.jsonl")


def test_expired_lease_is_recovered_and_attempt_is_incremented(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    first = TaskStore(path)
    second = TaskStore(path)
    first.enqueue("chunk", {"start": 0}, now=0.0)
    claimed = first.claim("dead-worker", lease_seconds=5, now=10.0)
    assert claimed is not None

    assert second.recover_expired(now=14.9) == 0
    assert second.recover_expired(now=15.0) == 1
    reclaimed = second.claim("replacement", lease_seconds=5, now=16.0)

    assert reclaimed is not None
    assert reclaimed.task_id == "chunk"
    assert reclaimed.owner == "replacement"
    assert reclaimed.attempt == 2
    with pytest.raises(LeaseLostError):
        first.complete("chunk", "dead-worker", now=16.0)


def test_expired_owner_cannot_heartbeat_or_complete(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "state.sqlite3")
    store.enqueue("chunk", {}, now=0.0)
    store.claim("worker", lease_seconds=5, now=1.0)

    with pytest.raises(LeaseLostError):
        store.heartbeat("chunk", "worker", lease_seconds=5, now=6.0)
    with pytest.raises(LeaseLostError):
        store.complete("chunk", "worker", now=6.0)


def test_failure_retry_and_permanent_failure(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "state.sqlite3")
    store.enqueue("retry", {"x": 1}, now=1.0)
    store.claim("w", lease_seconds=10, now=2.0)
    assert store.fail("retry", "w", "temporary", retry=True, now=3.0)
    assert store.get("retry").status == "pending"  # type: ignore[union-attr]

    claimed = store.claim("w2", lease_seconds=10, now=4.0)
    assert claimed is not None and claimed.attempt == 2
    assert store.fail("retry", "w2", "permanent", now=5.0)
    assert not store.fail("retry", "whoever", "permanent", now=6.0)
    assert store.get("retry").status == "failed"  # type: ignore[union-attr]

    assert store.requeue_failed("retry", now=7.0)
    assert not store.requeue_failed("retry", now=8.0)
    assert store.get("retry").status == "pending"  # type: ignore[union-attr]


def test_two_connections_cannot_claim_the_same_task(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    first = TaskStore(path)
    second = TaskStore(path)
    first.enqueue("only", {"chunk": 1})

    claimed = first.claim("one", lease_seconds=100)
    assert claimed is not None
    assert second.claim("two", lease_seconds=100) is None


def test_enqueue_many_is_atomic_on_conflict(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "state.sqlite3")
    with pytest.raises(TaskConflictError):
        store.enqueue_many(
            [("same", {"version": 1}), ("same", {"version": 2})], now=1.0
        )
    assert store.list_tasks() == ()


def test_attempt_token_fences_a_reused_owner_identity(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "state.sqlite3")
    store.enqueue("task", {}, now=0.0)
    first = store.claim("same-owner", lease_seconds=5, now=1.0)
    assert first is not None and first.attempt == 1
    store.recover_expired(now=6.0)
    second = store.claim("same-owner", lease_seconds=10, now=7.0)
    assert second is not None and second.attempt == 2

    with pytest.raises(LeaseLostError):
        store.heartbeat(
            "task", "same-owner", lease_seconds=10, attempt=first.attempt, now=8.0
        )
    with pytest.raises(LeaseLostError):
        store.complete("task", "same-owner", attempt=first.attempt, now=8.0)
    assert store.complete("task", "same-owner", attempt=second.attempt, now=8.0)


def test_completed_and_orphaned_tasks_have_explicit_recovery_transitions(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "state.sqlite3")
    store.enqueue_many((("done", {}), ("orphan", {})), now=0.0)
    done = store.claim("worker-a", lease_seconds=30, now=1.0)
    assert done is not None and done.task_id == "done"
    store.complete("done", "worker-a", result_path="lost.jsonl", now=2.0)
    orphan = store.claim("dead-owner", lease_seconds=30, now=3.0)
    assert orphan is not None and orphan.task_id == "orphan"

    assert store.requeue_done("done", now=4.0)
    assert store.get("done").result_path is None  # type: ignore[union-attr]
    assert store.requeue_orphaned_running("orphan", "dead-owner", now=4.0)
    assert store.counts()["pending"] == 2
