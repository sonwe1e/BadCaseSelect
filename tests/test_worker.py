from __future__ import annotations

import dataclasses
import threading
import time

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

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
from vfi_hard_miner.reconstruction import ReconstructionResult, pack_tier1_to_cpu
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


def test_easy_sample_fast_reject_skips_branch_diagnosis(monkeypatch):
    image = np.zeros((32, 40, 3), dtype=np.float32)
    flow = torch.zeros((1, 2, 32, 40), dtype=torch.float32)
    mask = torch.full((1, 1, 32, 40), 0.5, dtype=torch.float32)
    frame = torch.zeros((1, 3, 32, 40), dtype=torch.float32)
    reconstructed = ReconstructionResult(
        flow_t0=flow,
        flow_t1=flow,
        mask0=mask,
        mask1=mask,
        warp0=frame,
        warp1=frame,
        warp_blend=frame,
        prediction=frame,
    )

    def unexpected_diagnosis(*args, **kwargs):
        raise AssertionError("fast-reject must skip branch diagnosis")

    monkeypatch.setattr(worker_module, "diagnose_sample", unexpected_diagnosis)
    result = worker_module._sample_record(
        {
            "sample_id": "easy",
            "frame_indices": [1, 2, 3],
            "stride": 1,
        },
        img0=image,
        gt=image,
        img1=image,
        reconstructed=reconstructed,
        batch_index=0,
        thresholds=ThresholdConfig(),
    )

    assert result["status"] in {"invalid", "reject"}
    assert result["regions"] == []
    assert result["primary_region_index"] is None
    assert result["p_solvable"] == 0.0
    assert result["metrics"]["diagnosis"]["skipped"] == 1.0


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
    assert events[0][1][0][1].dtype == np.uint8
    assert events[1][1][0]["sample_id"] == "missing"
    assert events[2][1][0][0]["sample_id"] == "last"


def _overlapping_triplet_records(tmp_path):
    """Sliding window of shared frames: 0-1-2, 1-2-3, 2-3-4, ..."""

    frames = []
    for index in range(6):
        path = tmp_path / f"f{index}.png"
        _save(path, np.full((8, 8, 3), index * 10, dtype=np.uint8))
        frames.append(str(path))
    records = []
    for start in range(4):
        records.append(
            {
                "sample_id": f"t{start}",
                "img0": {"path": frames[start]},
                "gt": {"path": frames[start + 1]},
                "img1": {"path": frames[start + 2]},
            }
        )
    return records


def _drain(events):
    """Flatten decode events into a comparable (kind, sample_id, marker) list."""

    flat = []
    for kind, value in events:
        if kind == "batch":
            for record, first, _middle, _last in value.items:
                flat.append((kind, record["sample_id"], int(first[0, 0, 0])))
        else:
            flat.append((kind, value[0]["sample_id"], None))
    return flat


def test_decode_lookahead_matches_serial_event_stream(tmp_path):
    records = _overlapping_triplet_records(tmp_path)
    serial = _drain(
        _prefetched_decode_batches(
            records, batch_size=2, prefetch=2, max_cache=8, decode_workers=1
        )
    )
    parallel = _drain(
        _prefetched_decode_batches(
            records, batch_size=2, prefetch=2, max_cache=8, decode_workers=4
        )
    )
    assert parallel == serial
    # Sanity: overlapping triplets still group into the expected batches.
    assert [entry[0] for entry in serial] == ["batch"] * len(records)


def test_decode_lookahead_reads_each_unique_path_once(tmp_path, monkeypatch):
    records = _overlapping_triplet_records(tmp_path)
    unique_paths = {
        record[key]["path"] for record in records for key in ("img0", "gt", "img1")
    }
    reads: list[str] = []
    real_read = worker_module.read_rgb_uint8
    lock = threading.Lock()

    def counting_read(path):
        with lock:
            reads.append(path)
        return real_read(path)

    monkeypatch.setattr(worker_module, "read_rgb_uint8", counting_read)
    list(
        _prefetched_decode_batches(
            records, batch_size=2, prefetch=2, max_cache=64, decode_workers=4
        )
    )
    # With a cache large enough to hold the window, each shared frame decodes
    # exactly once even though adjacent triplets reference it repeatedly.
    assert sorted(reads) == sorted(unique_paths)
    assert len(reads) == len(unique_paths)


def test_decode_lookahead_preserves_invalid_order_with_workers(tmp_path):
    good = tmp_path / "good.png"
    _save(good, np.zeros((8, 8, 3), dtype=np.uint8))

    def record(sample_id, path):
        frame = {"path": str(path)}
        return {"sample_id": sample_id, "img0": frame, "gt": frame, "img1": frame}

    records = [
        record("a", good),
        record("bad", tmp_path / "nope.png"),
        record("b", good),
    ]
    events = list(
        _prefetched_decode_batches(
            records, batch_size=2, prefetch=2, max_cache=8, decode_workers=4
        )
    )
    assert [kind for kind, _ in events] == ["batch", "invalid", "batch"]
    assert events[0][1][0][0]["sample_id"] == "a"
    assert events[1][1][0]["sample_id"] == "bad"
    assert events[2][1][0][0]["sample_id"] == "b"


def test_network_input_batch_matches_legacy_full_resolution_resize():
    rng = np.random.default_rng(14)
    items = []
    for index in range(3):
        first = rng.integers(0, 256, size=(19, 27, 3), dtype=np.uint8)
        middle = rng.integers(0, 256, size=(19, 27, 3), dtype=np.uint8)
        last = rng.integers(0, 256, size=(19, 27, 3), dtype=np.uint8)
        items.append(({"sample_id": str(index)}, first, middle, last))

    actual0, actual1 = worker_module._network_input_batch(
        items,
        production_batch=4,
        network_size=(7, 11),
    )
    legacy0 = torch.stack(
        [
            worker_module._tensor_from_hwc(item[1]).to(torch.float32) / 255.0
            for item in (*items, items[-1])
        ]
    )
    legacy1 = torch.stack(
        [
            worker_module._tensor_from_hwc(item[3]).to(torch.float32) / 255.0
            for item in (*items, items[-1])
        ]
    )
    expected0 = F.interpolate(
        legacy0, size=(7, 11), mode="bilinear", align_corners=False
    )
    expected1 = F.interpolate(
        legacy1, size=(7, 11), mode="bilinear", align_corners=False
    )

    torch.testing.assert_close(actual0, expected0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual1, expected1, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("production_batch", [16, 32, 64])
def test_network_input_batch_keeps_only_network_resolution(production_batch):
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    items = [({"sample_id": "one"}, image, image, image)]

    img0, img1 = worker_module._network_input_batch(
        items,
        production_batch=production_batch,
        network_size=(18, 32),
    )

    assert img0.shape == img1.shape == (production_batch, 3, 18, 32)
    assert img0.numel() < production_batch * 3 * 180 * 320


def test_reconstruction_microbatch_converts_only_selected_uint8_items(monkeypatch):
    arrays = [
        np.full((8, 10, 3), value, dtype=np.uint8)
        for value in (10, 20, 30, 40, 50, 60)
    ]
    items = [
        ({"sample_id": "a"}, arrays[0], arrays[1], arrays[2]),
        ({"sample_id": "b"}, arrays[3], arrays[4], arrays[5]),
    ]
    converted = []
    original = worker_module.rgb_uint8_to_float32

    def counted(image):
        converted.append(id(image))
        return original(image)

    monkeypatch.setattr(worker_module, "rgb_uint8_to_float32", counted)
    prepared = worker_module._prepare_reconstruction_microbatch(items[:1])

    assert set(converted) == {id(value) for value in arrays[:3]}
    assert prepared.img0_tensor.shape == prepared.img1_tensor.shape == (1, 3, 8, 10)
    assert all(item.dtype == np.float32 for item in prepared.items[0][1:])


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

    def finish(items, reconstructed, *, config, tier2_residue=None):
        finish_threads.append(threading.current_thread().name)
        assert reconstructed.prediction.shape[0] == len(items)
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


# ---------------------------------------------------------------------------
# runtime.postproc_workers: resolution and executor propagation
# ---------------------------------------------------------------------------

import dataclasses  # noqa: E402


def test_resolve_postproc_workers_explicit_and_auto(monkeypatch):
    from vfi_hard_miner.worker import _resolve_postproc_workers

    def resolve(**runtime_overrides):
        config = _parallel_test_config()
        runtime = dataclasses.replace(config.runtime, **runtime_overrides)
        return _resolve_postproc_workers(dataclasses.replace(config, runtime=runtime))

    assert resolve(postproc_workers=5) == 5
    monkeypatch.setattr(worker_module.os, "cpu_count", lambda: 192)
    assert resolve(
        postproc_workers=0,
        cpu_threads_per_worker=1,
        workers=8,
    ) == 1
    assert resolve(
        postproc_workers=0,
        cpu_threads_per_worker=8,
        workers=8,
    ) == 2
    assert resolve(
        postproc_workers=0,
        cpu_threads_per_worker=128,
        workers=8,
    ) == 2


def test_negative_postproc_workers_fails_validation():
    runtime = RuntimeConfig(postproc_workers=-1)
    with pytest.raises(ValueError, match="postproc_workers"):
        runtime.validate()


def test_postproc_reservation_counts_reconstruction_inputs_and_scratch():
    image = np.zeros((10, 20, 3), dtype=np.float32)
    items = [({"sample_id": "a"}, image, image, image)]
    reservation = worker_module._postproc_reservation(
        items,
        buffer_bytes=16 * 1024 * 1024,
    )
    plane_bytes = 10 * 20 * 4

    assert worker_module.RECONSTRUCTION_CHANNELS == 18
    assert reservation.retained_bytes == plane_bytes * (18 + 9)
    assert reservation.scratch_bytes == plane_bytes * 24
    assert reservation.reconstruction_transient_bytes == plane_bytes * 6
    assert reservation.fixed_bytes == 1024 * 1024
    assert reservation.reserved_bytes == (
        reservation.retained_bytes
        + reservation.scratch_bytes
        + reservation.fixed_bytes
    )
    assert reservation.pipeline_bytes == (
        reservation.reserved_bytes
        + reservation.reconstruction_transient_bytes
    )
    assert reservation.oversize is False


def test_single_sample_over_budget_is_exclusive_microbatch():
    image = np.zeros((256, 256, 3), dtype=np.float32)
    items = [({"sample_id": "large"}, image, image, image)]

    reservation = worker_module._postproc_reservation(
        items,
        buffer_bytes=1 * 1024 * 1024,
    )
    microbatch = worker_module._postproc_microbatch_size(
        items,
        buffer_bytes=1 * 1024 * 1024,
        postproc_workers=2,
    )

    assert reservation.oversize is True
    assert microbatch == 1


def test_postproc_workers_propagates_to_executor(monkeypatch):
    config = _parallel_test_config(batch_size=2, prefetch=1)
    runtime = dataclasses.replace(config.runtime, postproc_workers=3)
    config = dataclasses.replace(config, runtime=runtime)
    created_workers = []
    real_executor = worker_module.ThreadPoolExecutor

    class _RecordingExecutor(real_executor):
        def __init__(self, max_workers=None, **kwargs):
            created_workers.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    item = _decoded_item("only", 0.25)
    item[0]["frame_indices"] = [1, 2, 3]
    item[0]["stride"] = 1
    batch = [item]

    def decoded_batches(*args, **kwargs):
        yield "batch", batch

    monkeypatch.setattr(worker_module, "_validate_payload_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_prefetched_decode_batches", decoded_batches)
    monkeypatch.setattr(worker_module, "ThreadPoolExecutor", _RecordingExecutor)

    result = process_main_payload(
        {"run_hash": config.run_hash(), "stage": "main", "triplets": []},
        adapter=_RecordingAdapter(),
        config=config,
    )

    assert created_workers == [3]
    assert [record["sample_id"] for record in result] == ["only"]


@pytest.mark.parametrize("record_count", [250, 256, 257])
def test_large_model_batch_is_split_into_memory_bounded_postproc_slices(
    monkeypatch,
    capsys,
    record_count,
):
    config = _parallel_test_config(batch_size=64, prefetch=1)
    runtime = dataclasses.replace(
        config.runtime,
        postproc_workers=2,
        postproc_buffer_mb=8,
    )
    config = dataclasses.replace(config, runtime=runtime)
    items = []
    for index in range(record_count):
        image = np.full((64, 64, 3), index / 1000, dtype=np.float32)
        items.append(({"sample_id": f"sample-{index}"}, image, image, image))
    batches = [items[start : start + 64] for start in range(0, record_count, 64)]
    reconstructed_sizes = []

    def decoded_batches(*args, **kwargs):
        for batch in batches:
            yield "batch", batch

    def reconstruct(img0, img1, outputs, **kwargs):
        batch = img0.shape[0]
        reconstructed_sizes.append(batch)
        flow = torch.zeros((batch, 2, 2, 2), dtype=torch.float32)
        mask = torch.zeros((batch, 1, 2, 2), dtype=torch.float32)
        image = torch.zeros((batch, 3, 2, 2), dtype=torch.float32)
        result = ReconstructionResult(
            flow_t0=flow,
            flow_t1=flow,
            mask0=mask,
            mask1=mask,
            warp0=image,
            warp1=image,
            warp_blend=image,
            prediction=image,
        )
        if kwargs.get("tiered"):
            return pack_tier1_to_cpu(result)
        return result

    def finish(batch, reconstructed, *, config, tier2_residue=None):
        return [{"sample_id": item[0]["sample_id"]} for item in batch]

    monkeypatch.setattr(worker_module, "_validate_payload_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_prefetched_decode_batches", decoded_batches)
    monkeypatch.setattr(worker_module, "_reconstruct_outputs", reconstruct)
    monkeypatch.setattr(worker_module, "_finish_main_batch", finish)

    result = process_main_payload(
        {
            "run_hash": config.run_hash(),
            "stage": "main",
            "triplets": [item[0] for item in items],
        },
        adapter=_RecordingAdapter(),
        config=config,
        progress_prefix="[test]",
    )

    assert [record["sample_id"] for record in result] == [
        f"sample-{index}" for index in range(record_count)
    ]
    assert sum(reconstructed_sizes) == record_count
    assert max(reconstructed_sizes) == 5
    progress = capsys.readouterr().err
    assert f"inferred {record_count}/{record_count}" in progress
    assert f"scored {record_count}/{record_count}" in progress
    assert "timing periodic" in progress
    assert "decode_ms/sample" in progress


def test_slow_postprocess_wait_emits_progress_and_heartbeat(monkeypatch, capsys):
    config = _parallel_test_config(batch_size=1, prefetch=1)
    runtime = dataclasses.replace(
        config.runtime,
        postproc_workers=1,
        postproc_buffer_mb=1,
    )
    config = dataclasses.replace(config, runtime=runtime)
    item = _decoded_item("slow", 0.5)
    heartbeats = []

    def decoded_batches(*args, **kwargs):
        yield "batch", [item]

    def slow_finish(batch, reconstructed, *, config, tier2_residue=None):
        threading.Event().wait(0.05)
        return [{"sample_id": "slow"}]

    monkeypatch.setattr(worker_module, "_validate_payload_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_prefetched_decode_batches", decoded_batches)
    monkeypatch.setattr(worker_module, "_finish_main_batch", slow_finish)
    monkeypatch.setattr(worker_module, "_FUTURE_WAIT_SECONDS", 0.005)
    monkeypatch.setattr(worker_module._ProgressLog, "_INTERVAL", 0.005)

    result = process_main_payload(
        {"run_hash": config.run_hash(), "stage": "main", "triplets": [item[0]]},
        adapter=_RecordingAdapter(),
        config=config,
        heartbeat=lambda: heartbeats.append(True),
        progress_prefix="[slow]",
    )

    assert result == [{"sample_id": "slow"}]
    assert len(heartbeats) >= 2
    progress = capsys.readouterr().err
    assert "pending 1 batches" in progress
    assert "retained" in progress
    assert "reserved" in progress
    assert "decode_uint8" in progress
    assert "resident_estimate" in progress
    assert "future_wait_ms/sample" in progress
    assert "reconstruction_ms/sample" in progress
    assert "scored 1/1" in progress


def test_decode_prefetch_respects_cache_budget_and_still_yields_batches(tmp_path):
    image_path = tmp_path / "frame.png"
    _save(image_path, np.zeros((8, 8, 3), dtype=np.uint8))

    def record(sample_id):
        frame = {"path": str(image_path)}
        return {"sample_id": sample_id, "img0": frame, "gt": frame, "img1": frame}

    events = list(
        _prefetched_decode_batches(
            [record("a"), record("b")],
            batch_size=2,
            prefetch=1,
            max_cache=258,
            cache_budget_bytes=16 * 1024 * 1024,
        )
    )

    assert [kind for kind, _ in events] == ["batch"]
    batch = events[0][1]
    assert [item[0]["sample_id"] for item in batch] == ["a", "b"]
    assert batch[0][1].dtype == np.uint8
    assert batch.uint8_bytes == 8 * 8 * 3


def test_postproc_reservation_tiered_splits_cpu_and_device_retention():
    image = np.zeros((10, 20, 3), dtype=np.float32)
    items = [({"sample_id": "a"}, image, image, image)]
    plane_bytes = 10 * 20 * 4

    tiered = worker_module._postproc_reservation(
        items, buffer_bytes=16 * 1024 * 1024, tiered=True
    )
    assert tiered.retained_bytes == plane_bytes * (7 + 9)
    assert tiered.device_retained_bytes == plane_bytes * 11
    assert tiered.scratch_bytes == plane_bytes * 24
    assert tiered.pipeline_bytes == (
        tiered.reserved_bytes
        + tiered.reconstruction_transient_bytes
        + tiered.device_retained_bytes
    )

    legacy = worker_module._postproc_reservation(
        items, buffer_bytes=16 * 1024 * 1024
    )
    assert legacy.retained_bytes == plane_bytes * (18 + 9)
    assert legacy.device_retained_bytes == 0
    assert legacy.pipeline_bytes == (
        legacy.reserved_bytes + legacy.reconstruction_transient_bytes
    )


def test_postproc_reservation_prediction_mode_retains_only_prediction():
    image = np.zeros((10, 20, 3), dtype=np.float32)
    items = [({"sample_id": "a"}, image, image, image)]
    plane_bytes = 10 * 20 * 4

    prediction = worker_module._postproc_reservation(
        items, buffer_bytes=16 * 1024 * 1024, mode="prediction"
    )
    assert prediction.retained_bytes == plane_bytes * (3 + 9)
    assert prediction.device_retained_bytes == 0
    assert prediction.pipeline_bytes == (
        prediction.reserved_bytes + prediction.reconstruction_transient_bytes
    )

    with pytest.raises(ValueError, match="mode"):
        worker_module._postproc_reservation(items, mode="bogus")


def test_teacher_prediction_only_path_never_builds_tier2_residue(tmp_path, monkeypatch):
    config, record, adapter = _endpoint_copy_fixture(tmp_path)
    teacher_model = ModelConfig(
        factory="vfi_hard_miner.mock_model:create_model",
        input_height=64,
        input_width=64,
        batch_size=2,
        factory_kwargs={"output_scale": 2},
    )
    config = dataclasses.replace(config, teacher=teacher_model)

    calls = []
    original_init = worker_module.Tier2Residue.__init__

    def spy_init(self, *args, **kwargs):
        calls.append(1)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(worker_module.Tier2Residue, "__init__", spy_init)

    results = worker_module._process_payload_records(
        [record],
        adapter=adapter,
        config=config,
        model_config=config.teacher,
        finish_batch=worker_module._finish_teacher_batch,
        invalid_record=lambda record, error: {
            "sample_id": record.get("sample_id"),
            "error": str(error),
        },
        heartbeat=None,
        progress_prefix="",
        reconstruction_device=None,
        thread_name_prefix="vfi-teacher-pred",
        prediction_only=True,
    )

    assert len(results) == 1
    # The prediction-only pack transfers 3ch and builds no tier-2 residue.
    assert calls == []
    assert "teacher" in results[0]


def _endpoint_copy_fixture(tmp_path, *, patches=(0, 128, 255)):
    """Three PNG frames; patch values control how wrong the prediction is."""

    root = tmp_path / "game"
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    first = base.copy()
    middle = base.copy()
    last = base.copy()
    first[20:44, 20:44] = patches[0]
    middle[20:44, 20:44] = patches[1]
    last[20:44, 20:44] = patches[2]
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
    adapter = ModelAdapter.from_config(config.model, device="cpu")
    return config, record, adapter


def test_tiered_pipeline_records_match_legacy_full_pack(tmp_path, monkeypatch):
    config, record, adapter = _endpoint_copy_fixture(tmp_path)
    monkeypatch.setattr(
        worker_module,
        "evaluate_in_scope",
        lambda *args, **kwargs: GateResult("review", ("needs_scope_review",), {}),
    )

    def run(tiered):
        return worker_module._process_payload_records(
            [record],
            adapter=adapter,
            config=config,
            model_config=config.model,
            finish_batch=worker_module._finish_main_batch,
            invalid_record=worker_module._invalid_record,
            heartbeat=None,
            progress_prefix="",
            reconstruction_device=None,
            thread_name_prefix="vfi-parity",
            tiered_reconstruction=tiered,
        )

    legacy = run(False)
    tiered = run(True)

    assert len(legacy) == 1
    # Sanity: the sample reached branch diagnosis (candidate path exercised).
    assert legacy[0]["p_wrong"] > 0.2
    assert "endpoint_copy" in legacy[0]["reasons"]
    # Bitwise record parity between the 18-channel pack and the two-tier path.
    assert tiered == legacy


def test_fast_rejected_samples_never_materialize_tier2(tmp_path, monkeypatch):
    # Identical frames: the prediction is correct, so every sample is
    # fast-rejected and the tier-2 warps/masks must never transfer.
    config, record, adapter = _endpoint_copy_fixture(tmp_path, patches=(128, 128, 128))
    calls = []
    original_materialize = worker_module.Tier2Residue.materialize

    def spy(self, indices=None):
        calls.append(id(self))
        return original_materialize(self, indices)

    monkeypatch.setattr(worker_module.Tier2Residue, "materialize", spy)
    results = worker_module._process_payload_records(
        [record],
        adapter=adapter,
        config=config,
        model_config=config.model,
        finish_batch=worker_module._finish_main_batch,
        invalid_record=worker_module._invalid_record,
        heartbeat=None,
        progress_prefix="",
        reconstruction_device=None,
        thread_name_prefix="vfi-spy",
        tiered_reconstruction=True,
    )

    assert len(results) == 1
    assert calls == []
    assert results[0]["metrics"]["diagnosis"]["skipped"] == 1.0


def test_pending_device_bytes_count_toward_the_postproc_budget(tmp_path):
    # 256x256 frames make the per-future tier-2 device retention (11ch,
    # 2.75 MiB) large enough to matter: with a 28 MiB budget, CPU-only
    # accounting admits two in-flight futures (reserved 11 MiB + next
    # pipeline 15.25 MiB = 26.25 MiB <= 28 MiB), but cumulative accounting
    # including the pending device bytes does not (26.25 + 2.75 = 29 MiB).
    root = tmp_path / "game"
    for index in range(1, 5):
        frame = np.full((256, 256, 3), 16 * index, dtype=np.uint8)
        _save(root / f"010000{index}.png", frame)
    config = AppConfig(
        data=DataConfig(root=str(root)),
        model=ModelConfig(
            factory="vfi_hard_miner.mock_model:create_model",
            input_height=64,
            input_width=64,
            batch_size=1,
            factory_kwargs={"output_scale": 2},
        ),
        runtime=RuntimeConfig(
            backend="cpu",
            devices=(0,),
            workers=1,
            state_db=str(tmp_path / "state.sqlite3"),
            run_dir=str(tmp_path / "run"),
            postproc_workers=4,
            postproc_buffer_mb=28,
        ),
    )
    build_run_index(config)
    triplets = build_index(root, frame_regex=config.data.frame_regex)
    assert len(triplets) == 2
    records = [
        serialize_triplet(triplet, run_hash=config.run_hash())
        for triplet in triplets
    ]
    adapter = ModelAdapter.from_config(config.model, device="cpu")

    state = {"running": 0, "peak": 0}
    lock = threading.Lock()

    def finish(items, reconstructed, *, config, tier2_residue=None):
        with lock:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        time.sleep(0.5)
        with lock:
            state["running"] -= 1
        return [{"sample_id": str(item[0]["sample_id"])} for item in items]

    results = worker_module._process_payload_records(
        records,
        adapter=adapter,
        config=config,
        model_config=config.model,
        finish_batch=finish,
        invalid_record=worker_module._invalid_record,
        heartbeat=None,
        progress_prefix="",
        reconstruction_device=None,
        thread_name_prefix="vfi-admission",
        tiered_reconstruction=True,
    )

    assert len(results) == 2
    # With cumulative device accounting the second future must wait for the
    # first to drain; CPU-only accounting would peak at two in-flight.
    assert state["peak"] == 1


def test_progress_log_reports_device_retained_and_tier2_fields(capsys):
    bar = worker_module._ProgressLog(10, "probe")
    bar.close(
        pending_batches=0,
        pending_bytes=2 * 1024 * 1024,
        pending_retained_bytes=1 * 1024 * 1024,
        pending_device_retained_bytes=4 * 1024 * 1024,
        tier2_d2h_bytes=8 * 1024 * 1024,
        tier2_materialized_batches=3,
        tier2_candidate_samples=7,
        memory=worker_module.MemoryEstimate(),
    )
    err = capsys.readouterr().err
    assert "device_retained 4 MiB" in err
    assert "tier2_d2h 8 MiB" in err
    assert "tier2_materialized_batches 3" in err
    assert "tier2_candidate_samples 7" in err
    # resident_estimate now covers pending CPU reserved + pending device.
    assert "resident_estimate 6 MiB" in err


def test_two_phase_materialize_runs_on_the_main_thread(tmp_path, monkeypatch):
    config, record, adapter = _endpoint_copy_fixture(tmp_path)
    monkeypatch.setattr(
        worker_module,
        "evaluate_in_scope",
        lambda *args, **kwargs: GateResult("review", ("needs_scope_review",), {}),
    )
    caller_thread = threading.current_thread()
    calls = []
    original_materialize = worker_module.Tier2Residue.materialize

    def spy(self, indices=None):
        calls.append(threading.current_thread())
        return original_materialize(self, indices)

    monkeypatch.setattr(worker_module.Tier2Residue, "materialize", spy)
    results = worker_module._process_payload_records(
        [record],
        adapter=adapter,
        config=config,
        model_config=config.model,
        finish_batch=worker_module._finish_main_batch,
        invalid_record=worker_module._invalid_record,
        heartbeat=None,
        progress_prefix="",
        reconstruction_device=None,
        thread_name_prefix="vfi-two-phase",
        tiered_reconstruction=True,
    )

    assert len(results) == 1
    # Candidate path exercised, and every tier-2 D2H happened on the
    # caller (main) thread — never on a postproc pool thread.
    assert calls
    assert all(thread is caller_thread for thread in calls)


def test_two_phase_splice_preserves_sample_order(tmp_path, monkeypatch):
    # A decode batch mixing a candidate (frames 1,2,3 change hard) with
    # fast-rejected samples (flat 255 frames): the two-phase splice must
    # restore the original sample order.
    root = tmp_path / "game"
    frames = [0, 128, 255, 255, 255]
    for index, value in enumerate(frames, start=1):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[16:48, 16:48] = value
        _save(root / f"01{index:05d}.png", image)
    config = AppConfig(
        data=DataConfig(root=str(root)),
        model=ModelConfig(
            factory="vfi_hard_miner.mock_model:create_model",
            input_height=64,
            input_width=64,
            batch_size=2,
            factory_kwargs={
                "output_scale": 2,
                "endpoint_copy_box": [0.25, 0.25, 0.75, 0.75],
            },
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
    triplets = build_index(root, frame_regex=config.data.frame_regex)
    assert len(triplets) == 3
    records = [
        serialize_triplet(triplet, run_hash=config.run_hash())
        for triplet in triplets
    ]
    adapter = ModelAdapter.from_config(config.model, device="cpu")

    results = worker_module._process_payload_records(
        records,
        adapter=adapter,
        config=config,
        model_config=config.model,
        finish_batch=worker_module._finish_main_batch,
        invalid_record=worker_module._invalid_record,
        heartbeat=None,
        progress_prefix="",
        reconstruction_device=None,
        thread_name_prefix="vfi-splice",
        tiered_reconstruction=True,
    )

    assert [r["sample_id"] for r in results] == [
        str(record["sample_id"]) for record in records
    ]
    diagnosed = [
        r for r in results if r["metrics"]["diagnosis"].get("skipped", 0.0) != 1.0
    ]
    skipped = [
        r for r in results if r["metrics"]["diagnosis"].get("skipped", 0.0) == 1.0
    ]
    assert len(diagnosed) + len(skipped) == 3
    # The hard-changing triplet is diagnosed; at least one flat triplet is
    # fast-rejected, so the batch exercises the mixed splice.
    assert len(diagnosed) >= 1
    assert len(skipped) >= 1


def test_two_phase_progress_reports_tier2_counters(tmp_path, monkeypatch, capsys):
    config, record, adapter = _endpoint_copy_fixture(tmp_path)
    monkeypatch.setattr(
        worker_module,
        "evaluate_in_scope",
        lambda *args, **kwargs: GateResult("review", ("needs_scope_review",), {}),
    )
    results = worker_module._process_payload_records(
        [record],
        adapter=adapter,
        config=config,
        model_config=config.model,
        finish_batch=worker_module._finish_main_batch,
        invalid_record=worker_module._invalid_record,
        heartbeat=None,
        progress_prefix="probe",
        reconstruction_device=None,
        thread_name_prefix="vfi-probe",
        tiered_reconstruction=True,
    )
    assert len(results) == 1

    done_lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("probe") and "done" in line
    ]
    assert done_lines
    final = done_lines[-1]
    assert "tier2_materialized_batches 1" in final
    assert "tier2_candidate_samples 1" in final
    assert "device_retained 0 MiB" in final
    # One candidate microbatch transferred its 11ch residue exactly once:
    # 64x64x11x4 = 176 KiB -> "tier2_d2h 0 MiB" at MiB granularity.
    assert "tier2_d2h 0 MiB" in final


def test_decode_workers_validation():
    assert RuntimeConfig().decode_workers == 1
    RuntimeConfig(decode_workers=4).validate()
    with pytest.raises(ValueError, match="decode_workers"):
        RuntimeConfig(decode_workers=0).validate()


def test_decode_workers_parallel_preserves_event_order(tmp_path):
    paths = {}
    for name, value in (
        ("a", 10),
        ("b", 20),
        ("c", 30),
        ("d", 40),
        ("e", 50),
        ("f", 60),
    ):
        frame_path = tmp_path / f"{name}.png"
        _save(frame_path, np.full((8, 8, 3), value, dtype=np.uint8))
        paths[name] = str(frame_path)

    def rec(sid, a, b, c):
        return {
            "sample_id": sid,
            "img0": {"path": a},
            "gt": {"path": b},
            "img1": {"path": c},
        }

    records = [
        rec("r1", paths["a"], paths["b"], paths["c"]),
        rec("bad", paths["d"], str(tmp_path / "missing.png"), paths["e"]),
        rec("r2", paths["e"], paths["f"], paths["a"]),
        rec("r3", paths["b"], paths["c"], paths["d"]),
    ]

    def run(workers):
        summary = []
        for kind, value in _prefetched_decode_batches(
            records,
            batch_size=2,
            prefetch=1,
            max_cache=64,
            decode_workers=workers,
        ):
            if kind == "batch":
                items = value.items
                summary.append(
                    ("batch", tuple(item[0]["sample_id"] for item in items))
                )
                summary.append(
                    ("pixels", tuple(int(item[1][0, 0, 0]) for item in items))
                )
            elif kind == "invalid":
                record, exc = value
                summary.append(("invalid", record["sample_id"], type(exc).__name__))
            else:
                summary.append((kind,))
        return summary

    serial = run(1)
    parallel = run(4)

    assert serial == [
        ("batch", ("r1",)),
        ("pixels", (10,)),
        ("invalid", "bad", "FileNotFoundError"),
        ("batch", ("r2", "r3")),
        ("pixels", (50, 20)),
    ]
    assert parallel == serial


def test_postproc_microbatch_size_override():
    image = np.zeros((64, 64, 3), dtype=np.float32)
    items = [({"sample_id": str(i)}, image, image, image) for i in range(64)]

    result = worker_module._postproc_microbatch_size(
        items,
        buffer_bytes=512 * 1024 * 1024,
        postproc_workers=4,
        override=8,
    )
    assert result == 8


def test_postproc_microbatch_size_override_caps_at_len():
    image = np.zeros((64, 64, 3), dtype=np.float32)
    items = [({"sample_id": str(i)}, image, image, image) for i in range(10)]

    result = worker_module._postproc_microbatch_size(
        items,
        buffer_bytes=512 * 1024 * 1024,
        postproc_workers=4,
        override=100,
    )
    assert result == 10


def test_postproc_microbatch_size_override_zero_uses_auto():
    image = np.zeros((64, 64, 3), dtype=np.float32)
    items = [({"sample_id": str(i)}, image, image, image) for i in range(64)]

    auto = worker_module._postproc_microbatch_size(
        items,
        buffer_bytes=512 * 1024 * 1024,
        postproc_workers=4,
    )
    with_zero = worker_module._postproc_microbatch_size(
        items,
        buffer_bytes=512 * 1024 * 1024,
        postproc_workers=4,
        override=0,
    )
    assert with_zero == auto
