from __future__ import annotations

import json
from pathlib import Path

import pytest

from vfi_hard_miner.offline import (
    BundleVerificationError,
    build_offline_archive,
    record_resource,
    verify_offline_archive,
    verify_resource_manifest,
)


def _manifest(root):
    third_party = root / "third_party"
    third_party.mkdir(parents=True, exist_ok=True)
    path = third_party / "manifest.json"
    path.write_text(
        json.dumps({"schema_version": 1, "target": {}, "resources": []}), encoding="utf-8"
    )
    return path


def test_record_and_verify_resource(tmp_path):
    manifest = _manifest(tmp_path)
    resource = tmp_path / "third_party" / "weights" / "model.bin"
    resource.parent.mkdir()
    resource.write_bytes(b"weights")
    record_resource(
        manifest,
        resource,
        bundle_root=tmp_path,
        name="model",
        kind="weight",
        source="local-test",
        version="1",
        license_name="test",
    )
    records = verify_resource_manifest(manifest, bundle_root=tmp_path)
    assert records[0].path == "third_party/weights/model.bin"


def test_verification_detects_tampering(tmp_path):
    manifest = _manifest(tmp_path)
    resource = tmp_path / "third_party" / "source.bin"
    resource.write_bytes(b"first")
    record_resource(
        manifest,
        resource,
        bundle_root=tmp_path,
        name="source",
        kind="source",
        source="local-test",
        version="1",
        license_name="test",
    )
    resource.write_bytes(b"changed")
    with pytest.raises(BundleVerificationError, match="mismatch"):
        verify_resource_manifest(manifest, bundle_root=tmp_path)


def _required_project(root: Path) -> Path:
    for directory in ("src", "scripts", "configs", "third_party", "ckpts/current"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "src" / "package.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "scripts" / "run.py").write_text("print('offline')\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='offline-test'\n", encoding="utf-8")
    (root / "requirements.lock").write_text("Pillow==12.0.0\n", encoding="utf-8")
    (root / "requirements-build.lock").write_text(
        "setuptools==80.9.0\n", encoding="utf-8"
    )
    (root / "README.md").write_text("offline test\n", encoding="utf-8")
    return _manifest(root)


def _add_generic_resource(root: Path, manifest: Path) -> Path:
    resource = root / "third_party" / "weights" / "metric.bin"
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_bytes(b"metric")
    record_resource(
        manifest,
        resource,
        bundle_root=root,
        name="metric",
        kind="weight",
        source="local-test",
        version="1",
        license_name="test",
    )
    return resource


def _add_checkpoint_config(root: Path, manifest: Path) -> Path:
    checkpoint = root / "ckpts" / "current" / "model.pth"
    checkpoint.write_bytes(b"current-model")
    (root / "configs" / "production.json").write_text(
        json.dumps({"model": {"checkpoint": "ckpts/current/model.pth"}}),
        encoding="utf-8",
    )
    record_resource(
        manifest,
        checkpoint,
        bundle_root=root,
        name="current-model",
        kind="weight",
        source="local-test",
        version="1",
        license_name="test",
    )
    return checkpoint


def test_production_bundle_rejects_empty_resource_manifest(tmp_path):
    root = tmp_path / "project"
    _required_project(root)
    with pytest.raises(BundleVerificationError, match="must not be empty"):
        build_offline_archive(root, tmp_path / "release.tar.gz")


def test_production_bundle_rejects_missing_required_root(tmp_path):
    root = tmp_path / "project"
    manifest = _required_project(root)
    _add_generic_resource(root, manifest)
    (root / "README.md").unlink()
    with pytest.raises(BundleVerificationError, match="missing=.*README.md"):
        build_offline_archive(root, tmp_path / "release.tar.gz")


def test_configured_checkpoint_must_exist_and_be_manifested(tmp_path):
    root = tmp_path / "project"
    manifest = _required_project(root)
    _add_generic_resource(root, manifest)
    config_path = root / "configs" / "production.json"
    config_path.write_text(
        json.dumps({"model": {"checkpoint": "ckpts/current/model.pth"}}),
        encoding="utf-8",
    )
    with pytest.raises(BundleVerificationError, match="checkpoint is missing"):
        build_offline_archive(root, tmp_path / "missing.tar.gz")

    (root / "ckpts" / "current" / "model.pth").write_bytes(b"unrecorded")
    with pytest.raises(BundleVerificationError, match="not listed"):
        build_offline_archive(root, tmp_path / "unrecorded.tar.gz")


def test_archive_destination_cannot_be_inside_an_included_root(tmp_path):
    root = tmp_path / "project"
    manifest = _required_project(root)
    _add_checkpoint_config(root, manifest)
    destination = root / "third_party" / "release.tar.gz"
    with pytest.raises(BundleVerificationError, match="outside every included root"):
        build_offline_archive(root, destination)
    assert not destination.exists()


def test_archive_inventory_covers_and_verifies_every_payload_file(tmp_path):
    root = tmp_path / "project"
    manifest = _required_project(root)
    checkpoint = _add_checkpoint_config(root, manifest)
    generated_cache = root / "src" / "__pycache__" / "package.cpython-312.pyc"
    generated_cache.parent.mkdir()
    generated_cache.write_bytes(b"generated")
    generated_metadata = root / "src" / "offline_test.egg-info" / "PKG-INFO"
    generated_metadata.parent.mkdir()
    generated_metadata.write_text("generated\n", encoding="utf-8")
    generated_dist_metadata = root / "src" / "offline_test.dist-info" / "METADATA"
    generated_dist_metadata.parent.mkdir()
    generated_dist_metadata.write_text("generated\n", encoding="utf-8")
    generated_mypy = root / "src" / ".mypy_cache" / "state.json"
    generated_mypy.parent.mkdir()
    generated_mypy.write_text("{}\n", encoding="utf-8")
    unrelated_checkpoint = root / "ckpts" / "current" / "private-old.pth"
    unrelated_checkpoint.write_bytes(b"must-not-leak")
    destination = tmp_path / "release.tar.gz"

    assert build_offline_archive(root, destination) == destination.resolve()
    inventory = verify_offline_archive(destination)
    by_path = {record.path: record for record in inventory}

    assert f"BadCaseSelect/{checkpoint.relative_to(root).as_posix()}" in by_path
    assert "BadCaseSelect/src/package.py" in by_path
    assert "BadCaseSelect/configs/production.json" in by_path
    assert "BadCaseSelect/third_party/manifest.json" in by_path
    assert "BadCaseSelect/src/__pycache__/package.cpython-312.pyc" not in by_path
    assert "BadCaseSelect/src/offline_test.egg-info/PKG-INFO" not in by_path
    assert "BadCaseSelect/src/offline_test.dist-info/METADATA" not in by_path
    assert "BadCaseSelect/src/.mypy_cache/state.json" not in by_path
    assert "BadCaseSelect/ckpts/current/private-old.pth" not in by_path
    assert all(record.size >= 0 and len(record.sha256) == 64 for record in inventory)


def test_resource_only_archive_does_not_require_user_checkpoint(tmp_path):
    root = tmp_path / "project"
    manifest = _required_project(root)
    resource = _add_generic_resource(root, manifest)
    (root / "configs" / "production.json").write_text(
        json.dumps({"model": {"checkpoint": "ckpts/current/model.pth"}}),
        encoding="utf-8",
    )
    extracted_demo = root / "third_party" / "src" / "metric" / "demo.mp4"
    extracted_demo.parent.mkdir(parents=True)
    extracted_demo.write_bytes(b"unregistered-demo")
    stale_wheel = root / "third_party" / "wheelhouse" / "stale.whl"
    stale_wheel.parent.mkdir(parents=True)
    stale_wheel.write_bytes(b"unregistered-wheel")
    destination = tmp_path / "resources.tar.gz"

    assert build_offline_archive(root, destination, resources_only=True) == destination.resolve()
    inventory = verify_offline_archive(destination)
    paths = {record.path for record in inventory}

    assert f"BadCaseSelect/{resource.relative_to(root).as_posix()}" in paths
    assert "BadCaseSelect/third_party/manifest.json" in paths
    assert "BadCaseSelect/requirements.lock" in paths
    assert "BadCaseSelect/requirements-build.lock" in paths
    assert "BadCaseSelect/src/package.py" not in paths
    assert "BadCaseSelect/configs/production.json" not in paths
    assert "BadCaseSelect/third_party/src/metric/demo.mp4" not in paths
    assert "BadCaseSelect/third_party/wheelhouse/stale.whl" not in paths


def test_archive_rejects_include_root_above_managed_resource_namespaces(tmp_path):
    root = tmp_path / "project"
    manifest = _required_project(root)
    _add_checkpoint_config(root, manifest)

    with pytest.raises(BundleVerificationError, match="ancestor of managed resource roots"):
        build_offline_archive(
            root,
            tmp_path / "unsafe.tar.gz",
            include_roots=(".",),
            required_roots=(".",),
        )


def test_archive_rejects_manifest_resource_filtered_as_generated(tmp_path):
    root = tmp_path / "project"
    manifest = _required_project(root)
    resource = root / "third_party" / "build" / "must-not-disappear.bin"
    resource.parent.mkdir(parents=True)
    resource.write_bytes(b"registered")
    record_resource(
        manifest,
        resource,
        bundle_root=root,
        name="registered-generated-path",
        kind="source",
        source="local-test",
        version="1",
        license_name="test",
    )

    with pytest.raises(BundleVerificationError, match="resources were omitted"):
        build_offline_archive(root, tmp_path / "resources.tar.gz", resources_only=True)
