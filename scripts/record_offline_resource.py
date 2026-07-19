#!/usr/bin/env python3
"""Add or update a file entry in third_party/manifest.json."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vfi_hard_miner.offline import record_resource  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--name", required=True)
    parser.add_argument("--kind", required=True, choices=("wheel", "source", "weight", "license"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--license", required=True, dest="license_name")
    args = parser.parse_args()
    record = record_resource(
        args.project_root / "third_party" / "manifest.json",
        args.path,
        bundle_root=args.project_root,
        name=args.name,
        kind=args.kind,
        source=args.source,
        version=args.version,
        license_name=args.license_name,
    )
    print(f"recorded {record.path} {record.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
