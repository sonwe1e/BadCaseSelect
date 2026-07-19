from __future__ import annotations

import threading

import numpy as np
from PIL import Image
import pytest
import torch

import vfi_hard_miner.worker as worker_module
from vfi_hard_miner.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    RuntimeConfig,
    ThresholdConfig,
)
from vfi_hard_miner.gates import GateResult
from vfi_hard_miner.model_adapter import ModelAdapter, ModelOutputs
from vfi_hard_miner.pipeline import build_run_index, execution_id, serialize_triplet
from vfi_hard_miner.indexing import build_index
from vfi_hard_miner.reconstruction import ReconstructionResult
from vfi_hard_miner.worker import (
    _pack_outputs_to_cpu,
    _prefetched_decode_batches,
    _warmup_adapter,
    process_main_payload,
    process_teacher_payload,
)


def _save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def test_process_main_payload_finds_solvable_endpoint_copy(tmp_path, monkeypatch):
    root = tmp_path / "game"
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    first = base.copy()
    middle = base.copy()
    last = base.copy()
    first[20:44, 20:44] = 0
    middle[20:44, 20:44] = 128
    last[20:44, 20:44] = 255
    _save(root / "0100001.png", first)
    _save(root / "0100002.png", middle)
    _save(root / "0100003.png", last)
    config = AppConfig(
        data=DataConfig(root=str(root)),
        model=ModelConfig(
            factory="vfi_hard_miner.mock_model:create_model",
            input_height=64,
            input_width=64,
            batch_size=2,
            factory_kwargs={"output_scale": 2, "endpoint_copy_box": [0.25, 0.25, 0.75, 0.75]},
        ),
        runtime=RuntimeConfig(
            backend="cpu",
            devices=(0,),
            workers=1,
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
        ),
    )
    build_run_index(config)
    triplet = build_index(root, frame_regex=config.data.frame_regex)[0]
    record = serialize_triplet(triplet, run_hash=config.run_hash())
    payload = {
        "run_hash": config.run_hash(),
        "execution_id": execution_id(config),
        "stage": "main",
        "video_id": triplet.video_id,
        "chunk_index": 0,
        "triplets": [record],
    }
    monkeypatch.setattr(
        worker_module,
        "evaluate_in_scope",
        lambda *args, **kwargs: GateResult("review", ("needs_scope_review",), {}),
    )
    adapter = ModelAdapter.from_config(config.model, device="cpu")
    results = process_main_payload(payload, adapter=adapter, config=config)
    assert len(results) == 1
    assert results[0]["valid"] is True
    assert results[0]["validity_label"] == "accept"
    assert results[0]["in_scope_label"] == "review"
    assert results[0]["in_scope"] is None
    assert results[0]["p_wrong"] > 0.2
    assert results[0]["mining_p_wrong"] == pytest.approx(results[0]["p_wrong"])
    assert results[0]["metrics"]["decision"]["p_wrong"] == pytest.approx(
        results[0]["mining_p_wrong"]
    )
    assert results[0]["p_solvable"] > 0.5
    assert "endpoint_copy" in results[0]["reasons"]


def test_packed_output_transfer_truncates_tail_and_splits_on_cpu():
    batch = 4
    outputs = ModelOutputs(
        torch.full((batch, 2, 3, 5), 1.0, dtype=torch.float64),
        torch.full((batch, 2, 3, 5), 2.0, dtype=torch.float64),
        torch.full((batch, 1, 3, 5), 0.25, dtype=torch.float64),
        torch.full((batch, 1, 3, 5), 0.75, dtype=torch.float64),
    )

    copied = _pack_outputs_to_cpu(outputs, valid_count=2)

    for tensor, channels in (
        (copied.flow_t0, 2),
        (copied.flow_t1, 2),
        (copied.mask0, 1),
        (copied.mask1, 1),
    ):
        assert tensor.shape == (2, channels, 3, 5)
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float32
    torch.testing.assert_close(copied.flow_t1, torch.full_like(copied.flow_t1, 2.0))
    torch.testing.assert_close(copied.mask1, torch.full_like(copied.mask1, 0.75))


def test_warmup_uses_fixed_production_batch_and_requested_count():
    class CountingAdapter:
        def __init__(self):
            self.shapes = []

        def infer(self, img0, img1):
            self.shapes.append((tuple(img0.shape), tuple(img1.shape)))
            batch = img0.shape[0]
            return ModelOutputs(
                torch.zeros((batch, 2, 2, 3)),
                torch.zeros((batch, 2, 2, 3)),
                torch.full((batch, 1, 2, 3), 0.5),
                torch.full((batch, 1, 2, 3), 0.5),
            )

    adapter = CountingAdapter()
    config = ModelConfig(
        factory="unused:factory",
        input_height=8,
        input_width=12,
        batch_size=3,
    )

    _warmup_adapter(adapter, config, warmup_batches=2)

    assert adapter.shapes == [((3, 3, 8, 12), (3, 3, 8, 12))] * 2


def test_decode_prefetch_preserves_batch_and_invalid_record_order(tmp_path):
    image_path = tmp_path / "frame.png"
    _save(image_path, np.zeros((8, 8, 3), dtype=np.uint8))

    def record(sample_id, path):
        frame = {"path": str(path)}
        return {"sample_id": sample_id, "img0": frame, "gt": frame, "img1": frame}

    records = [
        record("first", image_path),
        record("missing", tmp_path / "missing.png"),
        record("last", image_path),
    ]

    events = list(
        _prefetched_decode_batches(
            records,
            batch_size=2,
            prefetch=1,
            max_cache=8,
        )
    )

    assert [kind for kind, _ in events] == ["batch", "invalid", "batch"]
    assert events[0][1][0][0]["sample_id"] == "first"
    assert events[1][1][0]["sample_id"] == "missing"
    assert events[2][1][0][0]["sample_id"] == "last"


def _decoded_item(sample_id, marker):
    image = np.full((6, 8, 3), marker, dtype=np.float32)
    return ({"sample_id": sample_id}, image, image, image)


class _RecordingAdapter:
    def __init__(self, on_infer=None):
        self.on_infer = on_infer
        self.input_markers = []
        self.infer_threads = []

    def infer(self, img0, img1):
        self.input_markers.append(img0[:, 0, 0, 0].tolist())
        self.infer_threads.append(threading.current_thread().name)
        if self.on_infer is not None:
            self.on_infer(len(self.input_markers))
        batch = img0.shape[0]
        return ModelOutputs(
            torch.zeros((batch, 2, 3, 4)),
            torch.zeros((batch, 2, 3, 4)),
            torch.full((batch, 1, 3, 4), 0.5),
            torch.full((batch, 1, 3, 4), 0.5),
        )


def _parallel_test_config(*, batch_size=2, prefetch=1):
    model = ModelConfig(
        factory="unused:factory",
        input_height=6,
        input_width=8,
        batch_size=batch_size,
    )
    return AppConfig(
        data=DataConfig(root="."),
        model=model,
        teacher=model,
        runtime=RuntimeConfig(prefetch=prefetch, warmup_batches=0),
    )


def test_infer_output_batch_pads_model_batch_but_returns_only_valid_tail():
    adapter = _RecordingAdapter()
    items = [_decoded_item("first", 0.1), _decoded_item("second", 0.2)]

    img0, img1, outputs = worker_module._infer_output_batch(
        items,
        adapter=adapter,
        production_batch=4,
    )

    assert np.allclose(adapter.input_markers[0], [0.1, 0.2, 0.2, 0.2])
    assert img0.shape == img1.shape == (2, 3, 6, 8)
    assert outputs.flow_t0.shape == (2, 2, 3, 4)
    assert outputs.flow_t1.shape == (2, 2, 3, 4)
    assert outputs.mask0.shape == (2, 1, 3, 4)
    assert outputs.mask1.shape == (2, 1, 3, 4)
    assert all(
        tensor.device.type == "cpu"
        for tensor in (
            img0,
            img1,
            outputs.flow_t0,
            outputs.flow_t1,
            outputs.mask0,
            outputs.mask1,
        )
    )


@pytest.mark.parametrize(
    ("stage", "process", "finish_name", "records_key", "thread_prefix"),
    [
        ("main", process_main_payload, "_finish_main_batch", "triplets", "vfi-main-cpu"),
        (
            "teacher",
            process_teacher_payload,
            "_finish_teacher_batch",
            "records",
            "vfi-teacher-cpu",
        ),
    ],
)
def test_inference_overlaps_cpu_finish_and_preserves_batch_order(
    monkeypatch,
    stage,
    process,
    finish_name,
    records_key,
    thread_prefix,
):
    config = _parallel_test_config(batch_size=2, prefetch=1)
    first_batch = [_decoded_item("first", 0.1), _decoded_item("second", 0.2)]
    tail_batch = [_decoded_item("tail", 0.3)]
    cpu_started = threading.Event()
    second_inference_started = threading.Event()
    caller_thread = threading.current_thread().name
    finish_threads = []

    def on_infer(call_index):
        assert threading.current_thread().name == caller_thread
        if call_index == 2:
            assert cpu_started.wait(timeout=2.0), "CPU finish did not overlap inference"
            second_inference_started.set()

    adapter = _RecordingAdapter(on_infer=on_infer)

    def decoded_batches(*args, **kwargs):
        yield "batch", first_batch
        yield "batch", tail_batch

    def finish(items, img0, img1, outputs, *, config):
        finish_threads.append(threading.current_thread().name)
        assert img0.shape[0] == img1.shape[0] == len(items)
        assert outputs.flow_t0.shape[0] == len(items)
        if items[0][0]["sample_id"] == "first":
            cpu_started.set()
            assert second_inference_started.wait(timeout=2.0)
        return [{"sample_id": item[0]["sample_id"]} for item in items]

    monkeypatch.setattr(worker_module, "_validate_payload_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_prefetched_decode_batches", decoded_batches)
    monkeypatch.setattr(worker_module, finish_name, finish)
    result = process(
        {
            "run_hash": config.run_hash(),
            "stage": stage,
            records_key: [],
        },
        adapter=adapter,
        config=config,
    )

    assert [record["sample_id"] for record in result] == ["first", "second", "tail"]
    assert np.allclose(adapter.input_markers[0], [0.1, 0.2])
    assert np.allclose(adapter.input_markers[1], [0.3, 0.3])
    assert adapter.infer_threads == [caller_thread, caller_thread]
    assert finish_threads
    assert all(name.startswith(thread_prefix) for name in finish_threads)


@pytest.mark.parametrize(
    ("stage", "process", "finish_name", "records_key"),
    [
        ("main", process_main_payload, "_finish_main_batch", "triplets"),
        ("teacher", process_teacher_payload, "_finish_teacher_batch", "records"),
    ],
)
def test_cpu_postprocess_future_exception_is_propagated(
    monkeypatch,
    stage,
    process,
    finish_name,
    records_key,
):
    config = _parallel_test_config(batch_size=2, prefetch=1)
    batch = [_decoded_item("broken", 0.4)]

    def decoded_batches(*args, **kwargs):
        yield "batch", batch

    def fail_postprocess(*args, **kwargs):
        raise RuntimeError(f"{stage} CPU postprocess failed")

    monkeypatch.setattr(worker_module, "_validate_payload_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_prefetched_decode_batches", decoded_batches)
    monkeypatch.setattr(worker_module, finish_name, fail_postprocess)

    with pytest.raises(RuntimeError, match=f"{stage} CPU postprocess failed"):
        process(
            {
                "run_hash": config.run_hash(),
                "stage": stage,
                records_key: [],
            },
            adapter=_RecordingAdapter(),
            config=config,
        )


@pytest.mark.parametrize(
    ("stage", "process", "finish_name", "records_key", "invalid_status"),
    [
        (
            "main",
            process_main_payload,
            "_finish_main_batch",
            "triplets",
            "invalid",
        ),
        (
            "teacher",
            process_teacher_payload,
            "_finish_teacher_batch",
            "records",
            "review",
        ),
    ],
)
def test_decode_error_barrier_preserves_completed_batch_order(
    monkeypatch,
    stage,
    process,
    finish_name,
    records_key,
    invalid_status,
):
    config = _parallel_test_config(batch_size=2, prefetch=1)
    first_batch = [_decoded_item("before", 0.1)]
    final_batch = [_decoded_item("after", 0.2)]

    def decoded_batches(*args, **kwargs):
        yield "batch", first_batch
        yield "invalid", ({"sample_id": "invalid"}, ValueError("decode failed"))
        yield "batch", final_batch

    def finish(items, *args, **kwargs):
        return [
            {"sample_id": item[0]["sample_id"], "status": "finished"}
            for item in items
        ]

    monkeypatch.setattr(worker_module, "_validate_payload_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_prefetched_decode_batches", decoded_batches)
    monkeypatch.setattr(worker_module, finish_name, finish)
    result = process(
        {
            "run_hash": config.run_hash(),
            "stage": stage,
            records_key: [],
        },
        adapter=_RecordingAdapter(),
        config=config,
    )

    assert [record["sample_id"] for record in result] == ["before", "invalid", "after"]
    assert result[1]["status"] == invalid_status


def _teacher_reconstruction(prediction):
    height, width = prediction.shape[:2]
    image = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0)
    flow = torch.zeros((1, 2, height, width), dtype=torch.float32)
    mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
    return ReconstructionResult(
        flow_t0=flow,
        flow_t1=flow,
        mask0=mask,
        mask1=mask,
        warp0=image,
        warp1=image,
        warp_blend=image,
        prediction=image,
    )


def _teacher_source(*, regions, p_wrong, p_solvable, reason):
    return {
        "sample_id": "sample",
        "status": "review",
        "validity_label": "accept",
        "in_scope_label": "accept",
        "valid": True,
        "in_scope": True,
        "p_wrong": p_wrong,
        "p_solvable": p_solvable,
        "reasons": ["missing_part", reason],
        "regions": regions,
        "primary_region_index": 0 if regions else None,
        "metrics": {
            "validity": {
                "decode_ok": True,
                "finite": True,
                "sequence_contiguous": True,
                "duplicate_distance": 0.10,
                "scene_cut_score": 0.10,
                "temporal_asymmetry": 0.10,
                "max_adjacent_difference": 0.10,
            },
            "scope": {
                "out_of_bounds_ratio": 0.01,
                "flow_discontinuity_ratio": 0.01,
                "foreground_large_motion_ratio": 0.01,
                "occlusion_ratio": 0.01,
                "unexplained_motion_ratio": 0.01,
                "background_motion": 0.10,
            },
            "diagnosis": {
                "selected_p_wrong": p_wrong,
                "selected_p_solvable": p_solvable,
            },
            "decision": {"p_wrong": p_wrong, "p_solvable": p_solvable},
        },
    }


def test_teacher_rechecks_all_regions_and_reselects_primary():
    gt = np.zeros((64, 64, 3), dtype=np.float32)
    gt[4:24, 4:24] = 1.0
    gt[40:60, 40:60] = 1.0
    teacher_prediction = np.zeros_like(gt)
    teacher_prediction[40:60, 40:60] = 1.0
    regions = [
        {
            "box": [4, 4, 24, 24],
            "p_wrong": 0.95,
            "p_solvable": 0.50,
            "reasons": ["missing_part"],
            "metrics": {
                "current_error": 0.95,
                "warp0_error": 0.95,
                "warp1_error": 0.95,
                "warp_blend_error": 0.95,
                "p_solvable": 0.50,
            },
        },
        {
            "box": [40, 40, 60, 60],
            "p_wrong": 0.80,
            "p_solvable": 0.10,
            "reasons": ["broken_structure"],
            "metrics": {
                "current_error": 0.80,
                "warp0_error": 0.01,
                "warp1_error": 0.80,
                "warp_blend_error": 0.80,
                "p_solvable": 0.10,
            },
        },
    ]
    source = _teacher_source(
        regions=regions,
        p_wrong=0.95,
        p_solvable=0.50,
        reason="solvability_gray_zone",
    )

    updated = worker_module._teacher_update_record(
        source,
        gt=gt,
        reconstructed=_teacher_reconstruction(teacher_prediction),
        batch_index=0,
        thresholds=ThresholdConfig(missing_metrics_to_review=False),
    )

    assert updated["primary_region_index"] == 1
    assert updated["p_wrong"] == 0.80
    assert updated["p_solvable"] == updated["regions"][1]["p_solvable"]
    assert updated["regions"][1]["p_solvable"] > updated["regions"][0]["p_solvable"]
    assert updated["regions"][0]["teacher"]["local_error"] > 0.8
    assert updated["regions"][1]["teacher"]["local_error"] < 0.05
    assert updated["regions"][1]["teacher"]["solvability"]["best_warp_error"] == 0.01
    assert updated["regions"][1]["metrics"]["p_solvable"] == updated["p_solvable"]
    assert updated["teacher"]["region"] == [40, 40, 60, 60]
    assert updated["status"] == "accept"
    assert "solvability_gray_zone" not in updated["reasons"]
    assert "solvability_low" not in updated["reasons"]
    assert updated["main_decision"] == {
        "status": "review",
        "p_wrong": 0.95,
        "mining_p_wrong": 0.95,
        "p_solvable": 0.50,
        "reasons": ["missing_part", "solvability_gray_zone"],
        "decision": {"p_wrong": 0.95, "p_solvable": 0.50},
    }
    assert updated["metrics"]["diagnosis"]["selected_p_wrong"] == 0.80
    assert updated["metrics"]["diagnosis"]["selected_p_solvable"] == updated[
        "p_solvable"
    ]


def test_teacher_without_regions_keeps_global_fallback():
    gt = np.zeros((32, 32, 3), dtype=np.float32)
    source = _teacher_source(
        regions=[],
        p_wrong=0.80,
        p_solvable=0.10,
        reason="solvability_low",
    )

    updated = worker_module._teacher_update_record(
        source,
        gt=gt,
        reconstructed=_teacher_reconstruction(gt),
        batch_index=0,
        thresholds=ThresholdConfig(missing_metrics_to_review=False),
    )

    assert updated["primary_region_index"] is None
    assert updated["p_wrong"] == 0.80
    assert updated["p_solvable"] > 0.8
    assert updated["teacher"]["region"] is None
    assert updated["teacher"]["local_error"] == 0.0
    assert updated["status"] == "accept"
    assert updated["reasons"] == ["missing_part"]
    assert updated["main_decision"]["status"] == "review"
    assert updated["main_decision"]["reasons"] == [
        "missing_part",
        "solvability_low",
    ]


def test_teacher_reselection_preserves_and_uses_region_priority():
    gt = np.zeros((64, 64, 3), dtype=np.float32)
    gt[2:18, 2:30] = 1.0
    gt[24:52, 29:35] = 0.70
    regions = [
        {
            "box": [2, 2, 30, 18],
            "p_wrong": 0.95,
            "p_solvable": 0.20,
            "reasons": ["edge_tearing"],
            "metrics": {
                "current_error": 0.95,
                "warp0_error": 0.95,
                "warp1_error": 0.95,
                "warp_blend_error": 0.95,
                "ui_likelihood": 0.90,
                "priority_weight": 0.25,
            },
        },
        {
            "box": [29, 24, 35, 52],
            "p_wrong": 0.70,
            "p_solvable": 0.20,
            "reasons": ["broken_structure"],
            "metrics": {
                "current_error": 0.70,
                "warp0_error": 0.70,
                "warp1_error": 0.70,
                "warp_blend_error": 0.70,
                "ui_likelihood": 0.0,
                "priority_weight": 1.0,
            },
        },
    ]
    source = _teacher_source(
        regions=regions,
        p_wrong=0.95,
        p_solvable=0.20,
        reason="solvability_low",
    )

    updated = worker_module._teacher_update_record(
        source,
        gt=gt,
        reconstructed=_teacher_reconstruction(gt),
        batch_index=0,
        thresholds=ThresholdConfig(missing_metrics_to_review=False),
    )

    assert updated["primary_region_index"] == 1
    assert updated["p_wrong"] == pytest.approx(0.70)
    assert updated["mining_p_wrong"] == pytest.approx(0.70)
    assert updated["regions"][0]["metrics"]["priority_weight"] == pytest.approx(0.25)
    assert updated["regions"][1]["metrics"]["priority_weight"] == pytest.approx(1.0)
    assert updated["metrics"]["decision"]["p_wrong"] == pytest.approx(
        updated["mining_p_wrong"]
    )


def test_worker_payloads_reject_another_execution_snapshot(tmp_path):
    root = tmp_path / "game"
    root.mkdir()
    model = ModelConfig(factory="vfi_hard_miner.mock_model:create_model")
    config = AppConfig(
        data=DataConfig(root=str(root)),
        model=model,
        teacher=model,
        runtime=RuntimeConfig(
            backend="cpu",
            devices=(0,),
            workers=1,
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
        ),
    )
    build_run_index(config)

    with pytest.raises(RuntimeError, match="another execution snapshot"):
        process_main_payload(
            {
                "run_hash": config.run_hash(),
                "execution_id": "stale-execution",
                "stage": "main",
                "triplets": [],
            },
            adapter=None,
            config=config,
        )
    with pytest.raises(RuntimeError, match="another execution snapshot"):
        process_teacher_payload(
            {
                "run_hash": config.run_hash(),
                "execution_id": "stale-execution",
                "stage": "teacher",
                "records": [],
            },
            adapter=None,
            config=config,
        )
