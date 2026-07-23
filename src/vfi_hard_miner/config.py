"""Strict JSON configuration for reproducible mining runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DataConfig:
    root: str
    frame_regex: str | None = r"^(?P<video>.*?)(?P<frame>\d{5})\.(?P<ext>png|jpg|jpeg)$"
    frame_digits: int | None = None
    stride: int = 1
    recursive: bool = True
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")
    excluded_dirs: tuple[str, ...] = (
        "extremely_hard_case",
        "extremely_hard_case_visualization",
        ".git",
    )

    def validate(self) -> None:
        if not self.root:
            raise ValueError("data.root must not be empty")
        if self.stride < 1:
            raise ValueError("data.stride must be >= 1")
        if (self.frame_regex is None) == (self.frame_digits is None):
            raise ValueError("configure exactly one of data.frame_regex or data.frame_digits")
        if self.frame_digits is not None and self.frame_digits < 1:
            raise ValueError("data.frame_digits must be >= 1")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    factory: str
    checkpoint: str | None = None
    input_height: int = 540
    input_width: int = 960
    batch_size: int = 1
    mask0_role: Literal["warp0_weight", "warp1_weight"] = "warp0_weight"
    align_corners: bool = False
    padding_mode: Literal["zeros", "border", "reflection"] = "border"
    flow_units: Literal["input_pixels"] = "input_pixels"
    output_order: tuple[str, ...] = ("flow_t0", "flow_t1", "mask0", "mask1")
    factory_kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        # Accept both models/ short names (bare identifiers, resolved by
        # load_factory) and explicit 'module:function' paths.
        factory = self.factory.strip()
        if ":" not in factory and not factory.isidentifier():
            raise ValueError(
                "model.factory must be a models/ short name or use 'module:function'"
            )
        if self.input_height < 1 or self.input_width < 1:
            raise ValueError("model input size must be positive")
        if self.batch_size < 1:
            raise ValueError("model.batch_size must be >= 1")
        if tuple(self.output_order) != ("flow_t0", "flow_t1", "mask0", "mask1"):
            if sorted(self.output_order) != ["flow_t0", "flow_t1", "mask0", "mask1"]:
                raise ValueError("model.output_order must contain each required output exactly once")


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    reject_scene_cut: float = 0.62
    reject_duplicate: float = 0.002
    reject_out_of_bounds: float = 0.72
    duplicate_review_at: float | None = None
    scene_cut_review_at: float | None = None
    temporal_asymmetry_review_at: float = 0.65
    temporal_asymmetry_reject_at: float = 0.90
    menu_transition_review_at: float = 0.40
    menu_transition_reject_at: float = 0.70
    out_of_bounds_review_at: float | None = None
    occlusion_review_at: float = 0.40
    occlusion_reject_at: float = 0.70
    foreground_motion_review_at: float = 0.35
    foreground_motion_reject_at: float = 0.65
    flow_discontinuity_review_at: float = 0.30
    flow_discontinuity_reject_at: float = 0.60
    unexplained_motion_review_at: float = 0.35
    unexplained_motion_reject_at: float = 0.70
    wrong_reject_below: float = 0.20
    wrong_accept_at: float = 0.45
    solvable_reject_below: float = 0.30
    solvable_accept_at: float = 0.55
    edge_threshold: float = 0.12
    candidate_quantile: float = 0.96
    min_region_pixels: int = 9
    max_regions: int = 8
    ui_border_fraction: float = 0.20
    ui_border_min_overlap: float = 0.50
    ui_static_threshold: float = 0.08
    ui_gt_edge_threshold: float = 0.12
    ui_edge_density_target: float = 0.12
    ui_priority_floor: float = 0.35
    ui_likelihood_threshold: float = 0.55
    non_ui_region_reserve: int = 1
    missing_metrics_to_review: bool = True

    def validate(self) -> None:
        probability_fields = (
            "reject_scene_cut",
            "reject_duplicate",
            "reject_out_of_bounds",
            "duplicate_review_at",
            "scene_cut_review_at",
            "temporal_asymmetry_review_at",
            "temporal_asymmetry_reject_at",
            "menu_transition_review_at",
            "menu_transition_reject_at",
            "out_of_bounds_review_at",
            "occlusion_review_at",
            "occlusion_reject_at",
            "foreground_motion_review_at",
            "foreground_motion_reject_at",
            "flow_discontinuity_review_at",
            "flow_discontinuity_reject_at",
            "unexplained_motion_review_at",
            "unexplained_motion_reject_at",
            "wrong_reject_below",
            "wrong_accept_at",
            "solvable_reject_below",
            "solvable_accept_at",
            "edge_threshold",
            "candidate_quantile",
            "ui_border_fraction",
            "ui_border_min_overlap",
            "ui_static_threshold",
            "ui_gt_edge_threshold",
            "ui_edge_density_target",
            "ui_priority_floor",
            "ui_likelihood_threshold",
        )
        for name in probability_fields:
            raw_value = getattr(self, name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"thresholds.{name} must be in [0,1]")
        duplicate_review = (
            max(0.006, self.reject_duplicate * 3.0)
            if self.duplicate_review_at is None
            else self.duplicate_review_at
        )
        scene_review = (
            min(0.45, self.reject_scene_cut * 0.75)
            if self.scene_cut_review_at is None
            else self.scene_cut_review_at
        )
        out_of_bounds_review = (
            min(0.45, self.reject_out_of_bounds * 0.65)
            if self.out_of_bounds_review_at is None
            else self.out_of_bounds_review_at
        )
        if self.reject_duplicate > duplicate_review:
            raise ValueError(
                "duplicate frame reject threshold must be <= review threshold"
            )
        ordered_gray_zones = (
            ("scene cut", scene_review, self.reject_scene_cut),
            (
                "temporal asymmetry",
                self.temporal_asymmetry_review_at,
                self.temporal_asymmetry_reject_at,
            ),
            (
                "menu transition",
                self.menu_transition_review_at,
                self.menu_transition_reject_at,
            ),
            ("out of bounds", out_of_bounds_review, self.reject_out_of_bounds),
            ("occlusion", self.occlusion_review_at, self.occlusion_reject_at),
            (
                "foreground motion",
                self.foreground_motion_review_at,
                self.foreground_motion_reject_at,
            ),
            (
                "flow discontinuity",
                self.flow_discontinuity_review_at,
                self.flow_discontinuity_reject_at,
            ),
            (
                "unexplained motion",
                self.unexplained_motion_review_at,
                self.unexplained_motion_reject_at,
            ),
        )
        for label, review_at, reject_at in ordered_gray_zones:
            if review_at > reject_at:
                raise ValueError(
                    f"{label} review threshold must be <= reject threshold"
                )
        if self.wrong_accept_at < self.wrong_reject_below:
            raise ValueError("wrong accept threshold must be >= reject threshold")
        if self.solvable_accept_at < self.solvable_reject_below:
            raise ValueError("solvable accept threshold must be >= reject threshold")
        if self.min_region_pixels < 1 or self.max_regions < 1:
            raise ValueError("region limits must be positive")
        if not 0 <= self.non_ui_region_reserve <= self.max_regions:
            raise ValueError(
                "thresholds.non_ui_region_reserve must be between 0 and max_regions"
            )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    backend: Literal["cpu", "cuda", "npu"] = "cpu"
    devices: tuple[int, ...] = (0,)
    workers: int = 1
    chunk_triplets: int = 256
    prefetch: int = 2
    decode_cache_mb: int = 512
    cpu_threads_per_worker: int = 1
    postproc_workers: int = 0
    warmup_batches: int = 1
    lease_seconds: int = 1800
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    reconstruction: Literal["auto", "device", "cpu"] = "auto"
    state_db: str = "runs/state.sqlite3"
    run_dir: str = "runs/default"

    def validate(self) -> None:
        if self.workers < 1 or self.chunk_triplets < 1 or self.prefetch < 1:
            raise ValueError("runtime worker/chunk/prefetch values must be positive")
        if self.decode_cache_mb < 16:
            raise ValueError("runtime.decode_cache_mb must be >= 16")
        if self.cpu_threads_per_worker < 1:
            raise ValueError("runtime.cpu_threads_per_worker must be >= 1")
        if self.postproc_workers < 0:
            raise ValueError("runtime.postproc_workers must be >= 0")
        if self.warmup_batches < 0:
            raise ValueError("runtime.warmup_batches must be >= 0")
        if self.lease_seconds < 30:
            raise ValueError("runtime.lease_seconds must be >= 30")
        if not self.devices:
            raise ValueError("runtime.devices must not be empty")
        if self.workers > len(self.devices) and self.backend != "cpu":
            raise ValueError("accelerator workers cannot exceed device count")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    hard_case_dir: str = "extremely_hard_case"
    visualization_dir: str = "extremely_hard_case_visualization"
    manifest_name: str = "hard_case_manifest.jsonl"
    link_mode: Literal["hardlink_then_copy", "copy"] = "hardlink_then_copy"
    layout: Literal["segment_relative", "preserve_relative", "flat"] = (
        "segment_relative"
    )
    save_review: bool = False
    visualization_width: int = 320

    def validate(self) -> None:
        for value in (self.hard_case_dir, self.visualization_dir, self.manifest_name):
            if not value or Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError("output paths must be non-empty relative paths")
        if self.visualization_width < 64:
            raise ValueError("output.visualization_width must be >= 64")


@dataclass(frozen=True, slots=True)
class AppConfig:
    data: DataConfig
    model: ModelConfig
    teacher: ModelConfig | None = None
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        if self.teacher is not None:
            self.teacher.validate()
        self.thresholds.validate()
        self.runtime.validate()
        self.output.validate()

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def run_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path} must be an array")
        item_type = args[0] if args else Any
        return tuple(_coerce(item, item_type, f"{path}[]") for item in value)
    if origin is Literal:
        if value not in args:
            raise ValueError(f"{path} must be one of {args}")
        return value
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{path} must be an object")
        return value
    if origin is None and isinstance(annotation, type) and is_dataclass(annotation):
        return _dataclass_from_dict(annotation, value, path)
    if origin is not None and type(None) in args:
        if value is None:
            return None
        non_none = next(arg for arg in args if arg is not type(None))
        return _coerce(value, non_none, path)
    return value


def _dataclass_from_dict(cls: type[T], payload: Any, path: str) -> T:
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must be an object")
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown keys at {path}: {', '.join(unknown)}")
    hints = get_type_hints(cls)
    kwargs = {
        name: _coerce(value, hints[name], f"{path}.{name}")
        for name, value in payload.items()
    }
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"invalid configuration at {path}: {exc}") from exc


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = _dataclass_from_dict(AppConfig, payload, "config")
    config.validate()
    return config
