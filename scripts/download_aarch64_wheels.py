#!/usr/bin/env python3
"""Download binary aarch64 wheels on an internet-connected preparation host."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=Path("requirements.lock"))
    parser.add_argument(
        "--destination", type=Path, default=Path("third_party/wheelhouse/linux-aarch64")
    )
    parser.add_argument("--python-version", required=True, help="CPython digits, for example 311")
    parser.add_argument("--abi", help="ABI tag; defaults to cp<PYTHON_VERSION>")
    parser.add_argument("--platform", default="manylinux2014_aarch64")
    parser.add_argument("--index-url")
    args = parser.parse_args()
    if not args.python_version.isdigit() or len(args.python_version) not in (2, 3):
        parser.error("--python-version must look like 39, 310, 311, or 312")
    args.destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--requirement",
        str(args.requirements),
        "--dest",
        str(args.destination),
        "--platform",
        args.platform,
        "--implementation",
        "cp",
        "--python-version",
        args.python_version,
        "--abi",
        args.abi or f"cp{args.python_version}",
        "--only-binary=:all:",
    ]
    if args.index_url:
        command.extend(("--index-url", args.index_url))
    print(" ".join(command))
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
