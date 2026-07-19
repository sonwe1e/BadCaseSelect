#!/usr/bin/env python3
"""Verify third-party resources and create the transfer archive."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vfi_hard_miner.offline import (  # noqa: E402
    build_offline_archive,
    verify_bundle_project,
    verify_resource_bundle_project,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Destination .tar.gz; omit for verify-only")
    parser.add_argument(
        "--resources-only",
        action="store_true",
        help="verify/package third-party files without requiring user checkpoints",
    )
    args = parser.parse_args()
    records = (
        verify_resource_bundle_project(args.project_root)
        if args.resources_only
        else verify_bundle_project(args.project_root)
    )
    print(f"verified {len(records)} offline resources")
    if args.output is not None:
        destination = build_offline_archive(
            args.project_root,
            args.output,
            resources_only=args.resources_only,
        )
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
