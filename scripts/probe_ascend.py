#!/usr/bin/env python3
"""Read-only Ascend environment probe.

This script never installs, upgrades, or changes CANN/torch/torch_npu.  Run it
as a separate process before starting production workers, for example::

    python scripts/probe_ascend.py --require-devices 8 --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from vfi_hard_miner.runtime import collect_runtime_info  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report CANN, torch_npu, OS, Python, and visible Ascend devices without modifying them."
    )
    parser.add_argument(
        "--require-devices",
        type=int,
        default=0,
        metavar="N",
        help="exit non-zero unless at least N NPU devices are visible (default: 0)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when torch_npu import or NPU availability checks fail",
    )
    parser.add_argument(
        "--skip-npu-smi",
        action="store_true",
        help="do not execute the read-only 'npu-smi info' command",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.require_devices < 0:
        raise SystemExit("--require-devices must be non-negative")

    info = collect_runtime_info(
        include_npu=True,
        include_npu_smi=not args.skip_npu_smi,
    )
    checks: list[dict[str, object]] = []
    npu = info.get("npu", {})
    available = bool(npu.get("available", False))
    device_count = int(npu.get("device_count", 0))

    if args.strict:
        checks.append(
            {
                "name": "torch_npu_available",
                "passed": available,
                "detail": npu.get("error") if not available else "available",
            }
        )
    if args.require_devices:
        checks.append(
            {
                "name": "minimum_npu_devices",
                "passed": device_count >= args.require_devices,
                "detail": f"required={args.require_devices}, detected={device_count}",
            }
        )
    info["checks"] = checks
    print(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(bool(check["passed"]) for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
