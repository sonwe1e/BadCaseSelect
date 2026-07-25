"""Golden-record regression harness for parity verification.

Runs the complete pipeline (main + teacher + CGVQM) with the deterministic
mock model over synthetic frames and dumps every JSONL artifact under the
run directory as canonical JSON.  Run it against two code versions (e.g. a
baseline git worktree via PYTHONPATH, then the working tree) and diff the
dumps: p_wrong/mining_p_wrong/p_solvable/reasons/metrics must be identical.

Usage: python scripts/golden_record.py <output-dump.json>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "tmp_golden"
DATA = BASE / "data"
RUN = BASE / "run"
STATE = BASE / "state.sqlite3"

# NOTE: do not touch sys.path here.  The code version under test is chosen
# by the caller: PYTHONPATH=<worktree>/src for a baseline worktree, or the
# ambient editable install for the working tree.

from vfi_hard_miner.config import (  # noqa: E402
    AppConfig,
    CGVQMConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    RuntimeConfig,
    ThresholdConfig,
)
from vfi_hard_miner.manifest import read_jsonl  # noqa: E402
from vfi_hard_miner.pipeline import (  # noqa: E402
    build_run_index,
    run_main_stage,
    run_teacher_stage,
)
from vfi_hard_miner.cgvqm_stage import run_cgvqm_stage  # noqa: E402


_FRAME_VALUES = (0, 64, 128, 192, 255, 192, 128, 64)


def _build_data() -> None:
    if DATA.exists():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    for index, value in enumerate(_FRAME_VALUES, start=1):
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        image[24:72, 24:72] = value
        Image.fromarray(image).save(DATA / f"01{index:05d}.png")


def _build_config() -> AppConfig:
    weights = REPO / "third_party" / "weights" / "cgvqm"
    model = ModelConfig(
        factory="vfi_hard_miner.mock_model:create_model",
        input_height=96,
        input_width=96,
        batch_size=2,
        factory_kwargs={
            "output_scale": 2,
            "endpoint_copy_box": [0.25, 0.25, 0.75, 0.75],
        },
    )
    return AppConfig(
        data=DataConfig(root=str(DATA)),
        model=model,
        teacher=model,
        cgvqm=CGVQMConfig(
            enabled=True,
            backbone_checkpoint=str(weights / "r3d_18-b3b3357e.pth"),
            calibration_checkpoint=str(weights / "cgvqm-2.pickle"),
            backend="cpu",
            clip_frames=3,
            crop_size=32,
            candidates_per_task=2,
            b_error_at=0.1,
            a_error_at=0.5,
            temporal_persistence_at=0.2,
            spatial_overlap_at=0.1,
        ),
        thresholds=ThresholdConfig(
            wrong_reject_below=0.10,
            wrong_accept_at=0.20,
            severe_wrong_accept_at=0.20,
            solvable_reject_below=0.20,
            solvable_accept_at=0.50,
            missing_metrics_to_review=False,
        ),
        runtime=RuntimeConfig(
            backend="cpu",
            devices=(0,),
            workers=1,
            chunk_triplets=2,
            warmup_batches=0,
            state_db=str(STATE),
            run_dir=str(RUN),
        ),
        output=OutputConfig(
            link_mode="copy",
            layout="graded_flat",
            materialize_strategy="per_video",
            visualization_width=96,
        ),
    )


def _normalize(value):
    """Replace volatile absolute paths so dumps from different run
    directories (or machines sharing a data root) stay comparable."""

    if isinstance(value, str):
        return (
            value.replace(str(BASE), "<BASE>").replace(str(REPO), "<REPO>")
        )
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _dump() -> dict[str, list]:
    artifacts: dict[str, list] = {}
    for path in sorted(RUN.rglob("*.jsonl")):
        relative = path.relative_to(RUN).as_posix()
        records = [_normalize(record) for record in read_jsonl(path)]
        records.sort(key=lambda record: str(record.get("sample_id", "")))
        artifacts[relative] = records
    return artifacts


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    _build_data()
    if RUN.exists():
        shutil.rmtree(RUN)
    for stale in STATE.parent.glob(STATE.name + "*"):
        stale.unlink()

    config = _build_config()
    config_path = BASE / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(json.loads(config.canonical_json()), indent=2),
        encoding="utf-8",
    )

    build_run_index(config)
    run_main_stage(config_path)
    run_teacher_stage(config_path)
    run_cgvqm_stage(config_path)

    out.write_text(
        json.dumps(_dump(), ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"golden dump written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
