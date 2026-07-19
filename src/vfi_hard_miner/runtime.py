"""Runtime discovery and safe per-device worker bootstrap helpers.

Importing this module never imports :mod:`torch_npu` and never selects an NPU.
NPU initialization is confined to :func:`configure_device` (which callers
must invoke inside a worker) or the explicitly requested environment probe.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import torch


Backend = Literal["cpu", "cuda", "npu"]


@dataclass(frozen=True)
class DeviceSpec:
    """A device description that is safe to create in the parent process."""

    backend: Backend
    index: int | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "cuda", "npu"}:
            raise ValueError(f"unsupported device backend: {self.backend!r}")
        if self.backend == "cpu" and self.index is not None:
            raise ValueError("CPU device must not have an index")
        if self.backend != "cpu" and (self.index is None or self.index < 0):
            raise ValueError(f"{self.backend} device requires a non-negative index")

    def __str__(self) -> str:
        return self.backend if self.index is None else f"{self.backend}:{self.index}"


def parse_device(value: str | DeviceSpec | torch.device) -> DeviceSpec:
    """Parse a CPU/CUDA/NPU label without touching accelerator runtimes."""

    if isinstance(value, DeviceSpec):
        return value
    label = str(value).strip().lower()
    if not label:
        raise ValueError("device label must not be empty")
    backend, separator, index_text = label.partition(":")
    if backend == "cpu":
        if separator:
            raise ValueError("use 'cpu', not an indexed CPU device")
        return DeviceSpec("cpu")
    if backend not in {"cuda", "npu"}:
        raise ValueError(f"device must be cpu, cuda:N, or npu:N; got {value!r}")
    index = 0 if not separator else _parse_device_index(index_text, value)
    return DeviceSpec(backend, index)


def _parse_device_index(index_text: str, original: object) -> int:
    try:
        index = int(index_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid device index in {original!r}") from exc
    if index < 0:
        raise ValueError(f"device index must be non-negative, got {index}")
    return index


def configure_device(value: str | DeviceSpec | torch.device) -> torch.device:
    """Select one device in the current process.

    For NPU this function lazily imports ``torch_npu``.  In multi-process
    operation it must therefore be called by the spawned worker, never by the
    scheduler/parent process.
    """

    spec = parse_device(value)
    if spec.backend == "cpu":
        return torch.device("cpu")
    assert spec.index is not None

    if spec.backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        count = torch.cuda.device_count()
        if spec.index >= count:
            raise RuntimeError(f"CUDA device {spec.index} does not exist; detected {count} devices")
        torch.cuda.set_device(spec.index)
        return torch.device("cuda", spec.index)

    try:
        importlib.import_module("torch_npu")
    except Exception as exc:  # import may fail because CANN shared libraries are absent
        raise RuntimeError(
            "NPU was requested but torch_npu could not be imported. "
            "Use the torch/torch_npu/CANN versions already paired on the target server."
        ) from exc
    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError("torch_npu imported but torch.npu is not registered")
    if hasattr(npu, "is_available") and not npu.is_available():
        raise RuntimeError("NPU was requested but torch.npu.is_available() is false")
    count = int(npu.device_count())
    if spec.index >= count:
        raise RuntimeError(f"NPU device {spec.index} does not exist; detected {count} devices")
    npu.set_device(f"npu:{spec.index}")
    return torch.device("npu", spec.index)


def get_spawn_context() -> mp.context.BaseContext:
    """Return an explicit spawn context without changing global MP state."""

    return mp.get_context("spawn")


def _device_worker_bootstrap(
    worker_index: int,
    device_label: str,
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    device = configure_device(device_label)
    target(worker_index, device, *args, **kwargs)


def spawn_device_workers(
    target: Callable[..., Any],
    devices: Sequence[str | DeviceSpec | torch.device],
    *,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    daemon: bool = False,
    join: bool = True,
) -> list[mp.Process]:
    """Spawn one process per device and configure the device inside it.

    ``target`` must be importable/pickleable and receives
    ``(worker_index, torch_device, *args, **kwargs)``.  The parent only parses
    device labels; it does not import ``torch_npu`` or initialize an NPU.
    """

    if not callable(target):
        raise TypeError("target must be callable")
    parsed = [parse_device(device) for device in devices]
    if not parsed:
        raise ValueError("at least one worker device is required")
    labels = [str(device) for device in parsed]
    if len(set(labels)) != len(labels):
        raise ValueError(f"worker devices must be unique, got {labels}")

    context = get_spawn_context()
    processes: list[mp.Process] = []
    for worker_index, label in enumerate(labels):
        process = context.Process(
            target=_device_worker_bootstrap,
            args=(worker_index, label, target, tuple(args), dict(kwargs or {})),
            daemon=daemon,
            name=f"vfi-worker-{worker_index}-{label.replace(':', '-')}",
        )
        process.start()
        processes.append(process)

    if join:
        failures: list[str] = []
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failures.append(f"{process.name}: exit code {process.exitcode}")
        if failures:
            raise RuntimeError("one or more device workers failed: " + "; ".join(failures))
    return processes


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _candidate_cann_version_files() -> list[Path]:
    candidates: list[Path] = []
    for variable in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(variable)
        if value:
            root = Path(value)
            candidates.extend(
                [
                    root / "version.cfg",
                    root / "version.info",
                    root / "compiler" / "version.info",
                    root / "ascend_toolkit_install.info",
                ]
            )
    candidates.extend(
        [
            Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
            Path("/usr/local/Ascend/ascend-toolkit/latest/version.info"),
            Path("/etc/ascend_install.info"),
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def detect_cann_version() -> dict[str, str] | None:
    """Read CANN version metadata without executing or modifying CANN."""

    for candidate in _candidate_cann_version_files():
        try:
            if not candidate.is_file():
                continue
            lines = [
                line.strip()
                for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        except OSError:
            continue
        if lines:
            return {"source": str(candidate), "value": " | ".join(lines[:12])}
    return None


def _run_npu_smi(timeout_seconds: float = 10.0) -> dict[str, Any]:
    executable = shutil.which("npu-smi")
    if executable is None:
        return {"available": False, "error": "npu-smi was not found on PATH"}
    try:
        completed = subprocess.run(
            [executable, "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"npu-smi failed: {exc}"}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-5000:],
    }


def _probe_npu() -> dict[str, Any]:
    result: dict[str, Any] = {
        "torch_npu_distribution": _distribution_version("torch-npu"),
        "available": False,
        "device_count": 0,
        "devices": [],
    }
    try:
        torch_npu = importlib.import_module("torch_npu")
        result["torch_npu_module"] = str(getattr(torch_npu, "__version__", "unknown"))
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise RuntimeError("torch.npu was not registered")
        result["available"] = bool(npu.is_available()) if hasattr(npu, "is_available") else True
        count = int(npu.device_count())
        result["device_count"] = count
        devices: list[dict[str, Any]] = []
        for index in range(count):
            item: dict[str, Any] = {"index": index}
            if hasattr(npu, "get_device_name"):
                try:
                    item["name"] = str(npu.get_device_name(index))
                except Exception as exc:
                    item["name_error"] = str(exc)
            devices.append(item)
        result["devices"] = devices
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def collect_runtime_info(
    *,
    include_npu: bool = False,
    include_npu_smi: bool = False,
) -> dict[str, Any]:
    """Collect JSON-serializable environment information.

    ``include_npu=False`` is safe for the scheduler/parent.  Setting it to true
    explicitly imports ``torch_npu`` and is intended for the standalone probe
    process or a worker smoke test.
    """

    info: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        },
        "cann": detect_cann_version(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
                "ASCEND_OPP_PATH",
                "PYTORCH_NPU_ALLOC_CONF",
            )
            if os.environ.get(key) is not None
        },
    }
    if include_npu:
        info["npu"] = _probe_npu()
    else:
        info["npu"] = {"probed": False}
    if include_npu_smi:
        info["npu_smi"] = _run_npu_smi()
    return info


def runtime_info_json(*, include_npu: bool = False, include_npu_smi: bool = False) -> str:
    """Return stable, human-readable JSON for logs and manifests."""

    return json.dumps(
        collect_runtime_info(include_npu=include_npu, include_npu_smi=include_npu_smi),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
