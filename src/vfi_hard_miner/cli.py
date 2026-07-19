"""Command-line interface for the offline hard-case miner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

from . import __version__
from .config import load_config
from .finalize import finalize_run
from .offline import (
    build_offline_archive,
    verify_bundle_project,
    verify_resource_bundle_project,
)
from .pipeline import (
    build_run_index,
    run_main_stage,
    run_teacher_stage,
    stage_counts,
)
from .runtime import collect_runtime_info


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))


def _config_parser(subparsers: Any, name: str, help_text: str) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vfi-hard-miner", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--traceback", action="store_true", help="show full exceptions")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _config_parser(subparsers, "index", "scan frames and enqueue main-stage chunks")
    _config_parser(subparsers, "mine", "run the current model on all pending chunks")
    _config_parser(subparsers, "teacher", "run optional teacher candidate refinement")
    _config_parser(subparsers, "finalize", "merge segments and write final artifacts")
    _config_parser(subparsers, "run", "run index, main, optional teacher, and finalize")
    status = _config_parser(subparsers, "status", "show durable task counts")
    status.add_argument("--stage", choices=("main", "teacher", "all"), default="all")

    probe = subparsers.add_parser("probe", help="report the local runtime without modifying it")
    probe.add_argument("--backend", choices=("cpu", "cuda", "npu"), default="cpu")
    probe.add_argument("--npu-smi", action="store_true")
    probe.add_argument("--output", type=Path)

    verify = subparsers.add_parser("verify-bundle", help="verify offline resource hashes")
    verify.add_argument("--project-root", type=Path, default=Path.cwd())
    verify.add_argument("--manifest", type=Path)
    verify.add_argument(
        "--resources-only",
        action="store_true",
        help="verify third-party hashes before user model checkpoints are available",
    )

    bundle = subparsers.add_parser("build-bundle", help="verify and create an offline tar.gz")
    bundle.add_argument("--project-root", type=Path, default=Path.cwd())
    bundle.add_argument("--output", type=Path, required=True)
    bundle.add_argument(
        "--resources-only",
        action="store_true",
        help="package third-party resources and lock files before user checkpoints exist",
    )
    return parser


def _dispatch(args: argparse.Namespace) -> Any:
    if args.command == "probe":
        information = collect_runtime_info(
            include_npu=args.backend == "npu",
            include_npu_smi=args.npu_smi,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(information, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return information
    if args.command == "verify-bundle":
        root = args.project_root.resolve()
        manifest = args.manifest or root / "third_party" / "manifest.json"
        records = (
            verify_resource_bundle_project(root, manifest_path=manifest)
            if args.resources_only
            else verify_bundle_project(root, manifest_path=manifest)
        )
        return {"verified": len(records), "manifest": manifest}
    if args.command == "build-bundle":
        destination = build_offline_archive(
            args.project_root,
            args.output,
            resources_only=args.resources_only,
        )
        return {"archive": destination, "resources_only": args.resources_only}

    config = load_config(args.config)
    if args.command == "index":
        return build_run_index(config)
    if args.command == "mine":
        return run_main_stage(args.config)
    if args.command == "teacher":
        return run_teacher_stage(args.config)
    if args.command == "finalize":
        return finalize_run(args.config)
    if args.command == "status":
        stages = ("main", "teacher") if args.stage == "all" else (args.stage,)
        return {stage: stage_counts(config, stage=stage) for stage in stages}
    if args.command == "run":
        result: dict[str, Any] = {"index": build_run_index(config)}
        result["main"] = run_main_stage(args.config)
        if config.teacher is not None:
            result["teacher"] = run_teacher_stage(args.config)
        result["finalize"] = finalize_run(args.config)
        return result
    raise AssertionError(args.command)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except Exception as exc:
        if args.traceback:
            traceback.print_exc()
        else:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
