"""SQLite-backed durable task leases for independent offline workers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
from threading import Event, Thread
import time
from typing import Any, Iterable, Iterator, Mapping


TASK_STATUSES = frozenset({"pending", "running", "done", "failed"})


class TaskStateError(RuntimeError):
    """Base class for persistent task-state errors."""


class TaskConflictError(TaskStateError):
    """The same task ID was registered with different immutable content."""


class LeaseLostError(TaskStateError):
    """A worker attempted to mutate a task whose lease it no longer owns."""


class LeaseHeartbeat:
    """Renew one task lease on a dedicated SQLite connection."""

    def __init__(
        self,
        state_path: str | Path,
        task_id: str,
        owner: str,
        *,
        lease_seconds: float,
        attempt: int,
    ) -> None:
        self.state_path = Path(state_path)
        self.task_id = task_id
        self.owner = owner
        self.lease_seconds = float(lease_seconds)
        self.attempt = int(attempt)
        self._stop = Event()
        self._error: BaseException | None = None
        self._thread = Thread(
            target=self._run,
            name=f"vfi-lease-{task_id.rsplit(':', 1)[-1][:10]}",
            daemon=True,
        )

    def _run(self) -> None:
        interval = max(1.0, min(60.0, self.lease_seconds / 3.0))
        try:
            with TaskStore(self.state_path) as store:
                while not self._stop.wait(interval):
                    store.heartbeat(
                        self.task_id,
                        self.owner,
                        lease_seconds=self.lease_seconds,
                        attempt=self.attempt,
                    )
        except BaseException as exc:  # propagated by check() on the worker thread
            self._error = exc
            self._stop.set()

    def start(self) -> "LeaseHeartbeat":
        self._thread.start()
        return self

    def check(self) -> None:
        if self._error is not None:
            raise LeaseLostError(
                f"lease heartbeat failed for {self.task_id}: {self._error}"
            ) from self._error

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> "LeaseHeartbeat":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    payload: dict[str, Any]
    status: str
    priority: int
    owner: str | None
    lease_expires: float | None
    attempt: int
    result_path: str | None
    error: str | None
    created_at: float
    updated_at: float


def _canonical_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class TaskStore:
    """A small, process-safe queue implemented with short SQLite transactions.

    Construct one instance per spawned worker.  The class also detects an
    accidental post-fork use and transparently reopens the connection in the
    child rather than sharing SQLite connection state.
    """

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 30_000) -> None:
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must not be negative")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = busy_timeout_ms
        self._pid = -1
        self._connection: sqlite3.Connection | None = None
        self._connect()
        self._initialise_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            self._connection.close()
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        self._connection = connection
        self._pid = os.getpid()
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None or self._pid != os.getpid():
            return self._connect()
        return self._connection

    def _initialise_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending','running','done','failed')),
                priority INTEGER NOT NULL DEFAULT 0,
                owner TEXT,
                lease_expires REAL,
                attempt INTEGER NOT NULL DEFAULT 0 CHECK(attempt >= 0),
                result_path TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_claim
                ON tasks(status, priority DESC, created_at, task_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_lease
                ON tasks(status, lease_expires);
            """
        )

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"],
            payload=json.loads(row["payload_json"]),
            status=row["status"],
            priority=row["priority"],
            owner=row["owner"],
            lease_expires=row["lease_expires"],
            attempt=row["attempt"],
            result_path=row["result_path"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def enqueue(
        self,
        task_id: str,
        payload: Mapping[str, Any],
        *,
        priority: int = 0,
        now: float | None = None,
    ) -> bool:
        """Insert a pending task; identical repeated inserts are no-ops."""

        if not task_id:
            raise ValueError("task_id must not be empty")
        timestamp = time.time() if now is None else float(now)
        payload_json = _canonical_payload(payload)
        with self._write_transaction() as connection:
            row = connection.execute(
                "SELECT payload_json, priority FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is not None:
                if row["payload_json"] != payload_json or row["priority"] != priority:
                    raise TaskConflictError(
                        f"task {task_id!r} already exists with different content"
                    )
                return False
            connection.execute(
                """
                INSERT INTO tasks(
                    task_id, payload_json, status, priority, created_at, updated_at
                ) VALUES (?, ?, 'pending', ?, ?, ?)
                """,
                (task_id, payload_json, priority, timestamp, timestamp),
            )
            return True

    def enqueue_many(
        self,
        tasks: Iterable[tuple[str, Mapping[str, Any]]],
        *,
        priority: int = 0,
        now: float | None = None,
    ) -> int:
        timestamp = time.time() if now is None else float(now)
        inserted = 0
        with self._write_transaction() as connection:
            for task_id, payload in tasks:
                if not task_id:
                    raise ValueError("task_id must not be empty")
                payload_json = _canonical_payload(payload)
                row = connection.execute(
                    "SELECT payload_json, priority FROM tasks WHERE task_id=?", (task_id,)
                ).fetchone()
                if row is not None:
                    if row["payload_json"] != payload_json or row["priority"] != priority:
                        raise TaskConflictError(
                            f"task {task_id!r} already exists with different content"
                        )
                    continue
                connection.execute(
                    """
                    INSERT INTO tasks(
                        task_id, payload_json, status, priority, created_at, updated_at
                    ) VALUES (?, ?, 'pending', ?, ?, ?)
                    """,
                    (task_id, payload_json, priority, timestamp, timestamp),
                )
                inserted += 1
        return inserted

    def recover_expired(self, *, now: float | None = None) -> int:
        """Return expired running tasks to pending and clear their old owner."""

        timestamp = time.time() if now is None else float(now)
        with self._write_transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks
                   SET status='pending', owner=NULL, lease_expires=NULL,
                       updated_at=?
                 WHERE status='running'
                   AND lease_expires IS NOT NULL
                   AND lease_expires <= ?
                """,
                (timestamp, timestamp),
            )
            return cursor.rowcount

    def claim(
        self,
        owner: str,
        *,
        lease_seconds: float,
        now: float | None = None,
    ) -> TaskRecord | None:
        """Atomically recover expired leases and claim one highest-priority task."""

        if not owner:
            raise ValueError("owner must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = time.time() if now is None else float(now)
        lease_expires = timestamp + float(lease_seconds)
        with self._write_transaction() as connection:
            connection.execute(
                """
                UPDATE tasks
                   SET status='pending', owner=NULL, lease_expires=NULL,
                       updated_at=?
                 WHERE status='running'
                   AND lease_expires IS NOT NULL
                   AND lease_expires <= ?
                """,
                (timestamp, timestamp),
            )
            row = connection.execute(
                """
                SELECT task_id
                  FROM tasks
                 WHERE status='pending'
                 ORDER BY priority DESC, created_at ASC, task_id ASC
                 LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            task_id = row["task_id"]
            cursor = connection.execute(
                """
                UPDATE tasks
                   SET status='running', owner=?, lease_expires=?,
                       attempt=attempt+1, error=NULL, updated_at=?
                 WHERE task_id=? AND status='pending'
                """,
                (owner, lease_expires, timestamp, task_id),
            )
            if cursor.rowcount != 1:  # Defensive; BEGIN IMMEDIATE should prevent it.
                raise TaskStateError(f"failed to claim pending task {task_id!r}")
            claimed = connection.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            assert claimed is not None
            return self._row_to_record(claimed)

    def heartbeat(
        self,
        task_id: str,
        owner: str,
        *,
        lease_seconds: float,
        attempt: int | None = None,
        now: float | None = None,
    ) -> float:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = time.time() if now is None else float(now)
        lease_expires = timestamp + float(lease_seconds)
        with self._write_transaction() as connection:
            if attempt is None:
                cursor = connection.execute(
                    """
                    UPDATE tasks
                       SET lease_expires=?, updated_at=?
                     WHERE task_id=? AND status='running' AND owner=?
                       AND lease_expires > ?
                    """,
                    (lease_expires, timestamp, task_id, owner, timestamp),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE tasks
                       SET lease_expires=?, updated_at=?
                     WHERE task_id=? AND status='running' AND owner=?
                       AND attempt=? AND lease_expires > ?
                    """,
                    (lease_expires, timestamp, task_id, owner, attempt, timestamp),
                )
            if cursor.rowcount != 1:
                raise LeaseLostError(
                    f"worker {owner!r} no longer owns a live lease for {task_id!r}"
                )
        return lease_expires

    def complete(
        self,
        task_id: str,
        owner: str,
        *,
        result_path: str | Path | None = None,
        attempt: int | None = None,
        now: float | None = None,
    ) -> bool:
        """Mark a task done; repeating the same completion is idempotent."""

        timestamp = time.time() if now is None else float(now)
        result = None if result_path is None else str(result_path)
        with self._write_transaction() as connection:
            row = connection.execute(
                "SELECT status, owner, lease_expires, attempt, result_path FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            if row["status"] == "done":
                if result is not None and row["result_path"] != result:
                    raise TaskConflictError(
                        f"task {task_id!r} was already completed with another result"
                    )
                return False
            if (
                row["status"] != "running"
                or row["owner"] != owner
                or (attempt is not None and row["attempt"] != attempt)
                or row["lease_expires"] is None
                or row["lease_expires"] <= timestamp
            ):
                raise LeaseLostError(
                    f"worker {owner!r} does not own running task {task_id!r}"
                )
            connection.execute(
                """
                UPDATE tasks
                   SET status='done', lease_expires=NULL, result_path=?,
                       error=NULL, updated_at=?
                 WHERE task_id=?
                """,
                (result, timestamp, task_id),
            )
            return True

    def fail(
        self,
        task_id: str,
        owner: str,
        error: str,
        *,
        retry: bool = False,
        attempt: int | None = None,
        now: float | None = None,
    ) -> bool:
        """Record a worker failure, optionally returning the task to pending."""

        timestamp = time.time() if now is None else float(now)
        target_status = "pending" if retry else "failed"
        with self._write_transaction() as connection:
            row = connection.execute(
                "SELECT status, owner, lease_expires, attempt, error FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            if row["status"] == "failed" and not retry and row["error"] == error:
                return False
            if (
                row["status"] != "running"
                or row["owner"] != owner
                or (attempt is not None and row["attempt"] != attempt)
                or row["lease_expires"] is None
                or row["lease_expires"] <= timestamp
            ):
                raise LeaseLostError(
                    f"worker {owner!r} does not own running task {task_id!r}"
                )
            next_owner = None if retry else owner
            connection.execute(
                """
                UPDATE tasks
                   SET status=?, owner=?, lease_expires=NULL, error=?, updated_at=?
                 WHERE task_id=?
                """,
                (target_status, next_owner, error, timestamp, task_id),
            )
            return True

    def requeue_failed(self, task_id: str, *, now: float | None = None) -> bool:
        timestamp = time.time() if now is None else float(now)
        with self._write_transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks
                   SET status='pending', owner=NULL, lease_expires=NULL,
                       error=NULL, updated_at=?
                 WHERE task_id=? AND status='failed'
                """,
                (timestamp, task_id),
            )
            return cursor.rowcount == 1

    def requeue_done(self, task_id: str, *, now: float | None = None) -> bool:
        """Re-run a completed task whose externally stored result was lost.

        Callers must validate that the result part/artifacts are absent or
        corrupt before invoking this recovery transition.
        """

        timestamp = time.time() if now is None else float(now)
        with self._write_transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks
                   SET status='pending', owner=NULL, lease_expires=NULL,
                       result_path=NULL, error=NULL, updated_at=?
                 WHERE task_id=? AND status='done'
                """,
                (timestamp, task_id),
            )
            return cursor.rowcount == 1

    def requeue_orphaned_running(
        self,
        task_id: str,
        owner: str,
        *,
        now: float | None = None,
    ) -> bool:
        """Requeue a running task only when the caller proved its owner died."""

        timestamp = time.time() if now is None else float(now)
        with self._write_transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks
                   SET status='pending', owner=NULL, lease_expires=NULL,
                       error='orphaned worker recovered', updated_at=?
                 WHERE task_id=? AND status='running' AND owner=?
                """,
                (timestamp, task_id, owner),
            )
            return cursor.rowcount == 1

    def get(self, task_id: str) -> TaskRecord | None:
        row = self.connection.execute(
            "SELECT * FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        return None if row is None else self._row_to_record(row)

    def list_tasks(self, *, status: str | None = None) -> tuple[TaskRecord, ...]:
        if status is not None and status not in TASK_STATUSES:
            raise ValueError(f"unknown task status: {status!r}")
        if status is None:
            rows = self.connection.execute(
                "SELECT * FROM tasks ORDER BY created_at, task_id"
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY created_at, task_id",
                (status,),
            ).fetchall()
        return tuple(self._row_to_record(row) for row in rows)

    def counts(self) -> dict[str, int]:
        counts = {status: 0 for status in sorted(TASK_STATUSES)}
        for row in self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
        ):
            counts[row["status"]] = row["count"]
        return counts

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "TaskStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# Backwards-friendly name for the component rather than the implementation.
TaskQueue = TaskStore
