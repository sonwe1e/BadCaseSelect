from __future__ import annotations

import json
from pathlib import Path

import pytest

from vfi_hard_miner.config import load_config
from vfi_hard_miner.gates import GateConfig
from vfi_hard_miner.scoring import ScoringConfig


def _payload(tmp_path):
    return {
        "data": {"root": str(tmp_path)},
        "model": {"factory": "examples.mock_model:create_model"},
    }


def test_config_loads_defaults_and_has_stable_hash(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_payload(tmp_path)), encoding="utf-8")
    first = load_config(path)
    second = load_config(path)
    assert first.model.input_height == 540
    assert first.data.extensions == (".png", ".jpg", ".jpeg")
    assert first.thresholds.missing_metrics_to_review is True
    assert GateConfig.from_value(first.thresholds) == GateConfig()
    assert first.runtime.cpu_threads_per_worker == 1
    assert first.runtime.warmup_batches == 1
    assert first.output.layout == "segment_relative"
    assert first.run_hash() == second.run_hash()


def test_config_rejects_unknown_keys(tmp_path):
    payload = _payload(tmp_path)
    payload["model"]["guess_mask_for_each_sample"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_config_exposes_scope_and_menu_gate_thresholds(tmp_path):
    payload = _payload(tmp_path)
    payload["thresholds"] = {
        "menu_transition_review_at": 0.21,
        "menu_transition_reject_at": 0.41,
        "out_of_bounds_review_at": 0.22,
        "occlusion_review_at": 0.23,
        "occlusion_reject_at": 0.43,
        "foreground_motion_review_at": 0.24,
        "foreground_motion_reject_at": 0.44,
        "flow_discontinuity_review_at": 0.25,
        "flow_discontinuity_reject_at": 0.45,
        "unexplained_motion_review_at": 0.26,
        "unexplained_motion_reject_at": 0.46,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_config(path)
    gates = GateConfig.from_value(config.thresholds)

    assert gates.menu_transition_review_at == 0.21
    assert gates.menu_transition_reject_at == 0.41
    assert gates.out_of_bounds_review_at == 0.22
    assert gates.occlusion_review_at == 0.23
    assert gates.occlusion_reject_at == 0.43
    assert gates.foreground_motion_review_at == 0.24
    assert gates.foreground_motion_reject_at == 0.44
    assert gates.flow_discontinuity_review_at == 0.25
    assert gates.flow_discontinuity_reject_at == 0.45
    assert gates.unexplained_motion_review_at == 0.26
    assert gates.unexplained_motion_reject_at == 0.46


def test_config_exposes_ui_priority_thresholds(tmp_path):
    payload = _payload(tmp_path)
    payload["thresholds"] = {
        "ui_border_fraction": 0.18,
        "ui_border_min_overlap": 0.62,
        "ui_static_threshold": 0.04,
        "ui_gt_edge_threshold": 0.15,
        "ui_edge_density_target": 0.20,
        "ui_priority_floor": 0.25,
        "ui_likelihood_threshold": 0.58,
        "non_ui_region_reserve": 2,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    scoring = ScoringConfig.from_value(load_config(path).thresholds)

    assert scoring.ui_border_fraction == 0.18
    assert scoring.ui_border_min_overlap == 0.62
    assert scoring.ui_static_threshold == 0.04
    assert scoring.ui_gt_edge_threshold == 0.15
    assert scoring.ui_edge_density_target == 0.20
    assert scoring.ui_priority_floor == 0.25
    assert scoring.ui_likelihood_threshold == 0.58
    assert scoring.non_ui_region_reserve == 2


def test_config_rejects_ui_reserve_larger_than_region_budget(tmp_path):
    payload = _payload(tmp_path)
    payload["thresholds"] = {"max_regions": 2, "non_ui_region_reserve": 3}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="non_ui_region_reserve"):
        load_config(path)


def test_legacy_threshold_names_still_drive_gate_config(tmp_path):
    payload = _payload(tmp_path)
    payload["thresholds"] = {
        "reject_duplicate": 0.001,
        "reject_scene_cut": 0.81,
        "reject_out_of_bounds": 0.91,
        "solvable_reject_below": 0.19,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    gates = GateConfig.from_value(load_config(path).thresholds)

    assert gates.duplicate_reject_at == 0.001
    assert gates.scene_cut_reject_at == 0.81
    assert gates.out_of_bounds_reject_at == 0.91
    assert gates.solvable_review_below == 0.19


def test_config_rejects_reversed_scope_gray_zone(tmp_path):
    payload = _payload(tmp_path)
    payload["thresholds"] = {
        "foreground_motion_review_at": 0.80,
        "foreground_motion_reject_at": 0.60,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="foreground motion"):
        load_config(path)


def test_config_rejects_ambiguous_factory(tmp_path):
    payload = _payload(tmp_path)
    # Neither a valid short-name identifier nor 'module:function' syntax.
    payload["model"]["factory"] = "not a factory"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="module:function"):
        load_config(path)


def test_config_accepts_short_name_factory(tmp_path):
    payload = _payload(tmp_path)
    payload["model"]["factory"] = "unet"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_config(path)
    assert config.model.factory == "unet"


def test_config_allows_fixed_frame_digits(tmp_path):
    payload = _payload(tmp_path)
    payload["data"]["frame_regex"] = None
    payload["data"]["frame_digits"] = 5
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_config(path).data.frame_digits == 5


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("cpu_threads_per_worker", 0, "cpu_threads_per_worker"),
        ("postproc_buffer_mb", 0, "postproc_buffer_mb"),
        ("warmup_batches", -1, "warmup_batches"),
    ],
)
def test_config_rejects_invalid_worker_runtime_values(tmp_path, name, value, match):
    payload = _payload(tmp_path)
    payload["runtime"] = {name: value}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_per_video_materialization_requires_segment_relative_without_teacher(tmp_path):
    payload = _payload(tmp_path)
    payload["output"] = {
        "materialize_strategy": "per_video",
        "layout": "flat",
    }
    path = tmp_path / "flat.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="segment_relative"):
        load_config(path)

    payload["output"]["layout"] = "segment_relative"
    payload["teacher"] = dict(payload["model"])
    path = tmp_path / "teacher.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not support teacher"):
        load_config(path)


def test_graded_flat_requires_cgvqm_and_byte_copy(tmp_path):
    payload = _payload(tmp_path)
    payload["output"] = {
        "layout": "graded_flat",
        "link_mode": "hardlink_then_copy",
    }
    path = tmp_path / "graded.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="requires output.link_mode='copy'"):
        load_config(path)

    payload["output"]["link_mode"] = "copy"
    with pytest.raises(ValueError, match="requires cgvqm.enabled=true"):
        path.write_text(json.dumps(payload), encoding="utf-8")
        load_config(path)

    payload["cgvqm"] = {
        "enabled": True,
        "spatial_overlap_at": 0.4,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_config(path)
    assert config.output.layout == "graded_flat"
    assert config.cgvqm.spatial_overlap_at == 0.4


_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_example_json_loads_with_production_settings():
    config = load_config(_CONFIGS_DIR / "example.json")
    assert config.runtime.prefetch == 4
    assert config.runtime.decode_workers == 4
    assert config.runtime.decode_cache_mb == 1024
    assert config.runtime.postproc_workers == 4
    assert config.runtime.postproc_buffer_mb == 2048
    assert config.runtime.reconstruction == "device"
    assert config.cgvqm.batch_size == 4
    assert config.cgvqm.context_cache_mb == 256
    # Calibration baseline stays put.
    assert config.model.batch_size == 64
    assert config.runtime.cpu_threads_per_worker == 8
    assert config.runtime.precision == "float32"


def test_example_jsonc_matches_example_json():
    def strip_line_comments(text: str) -> str:
        out = []
        in_string = False
        escaped = False
        index = 0
        while index < len(text):
            char = text[index]
            if in_string:
                out.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
                out.append(char)
            elif char == "/" and text[index + 1 : index + 2] == "/":
                newline = text.find("\n", index)
                if newline == -1:
                    break
                index = newline - 1
            else:
                out.append(char)
            index += 1
        return "".join(out)

    plain = json.loads((_CONFIGS_DIR / "example.json").read_text(encoding="utf-8"))
    annotated = json.loads(
        strip_line_comments(
            (_CONFIGS_DIR / "example.jsonc").read_text(encoding="utf-8")
        )
    )
    assert annotated == plain
