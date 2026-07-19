from __future__ import annotations

import json

from PIL import Image

from vfi_hard_miner.cli import main


def test_cli_probe_writes_json(capsys):
    assert main(["probe", "--backend", "cpu"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["python"]["version"]
    assert payload["npu"] == {"probed": False}


def test_cli_index(tmp_path, capsys):
    root = tmp_path / "game"
    root.mkdir()
    for index in range(1, 4):
        Image.new("RGB", (8, 8), (index, index, index)).save(root / f"01{index:05d}.png")
    config = {
        "data": {"root": str(root)},
        "model": {"factory": "vfi_hard_miner.mock_model:create_model"},
        "runtime": {
            "backend": "cpu",
            "devices": [0],
            "workers": 1,
            "state_db": str(tmp_path / "state.sqlite3"),
            "run_dir": str(tmp_path / "run"),
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert main(["index", "--config", str(config_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["triplets"] == 1
    assert output["inserted_tasks"] == 1


def test_cli_reports_config_error(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")
    assert main(["index", "--config", str(path)]) == 2
    assert "invalid configuration" in capsys.readouterr().err


def test_cli_resource_verification_checks_bundle_prerequisites(tmp_path, capsys):
    root = tmp_path / "project"
    third_party = root / "third_party"
    third_party.mkdir(parents=True)
    (third_party / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "target": {}, "resources": []}),
        encoding="utf-8",
    )
    (root / "requirements.lock").write_text("Pillow==11.3.0\n", encoding="utf-8")

    assert main(["verify-bundle", "--project-root", str(root), "--resources-only"]) == 2
    assert "requirements-build.lock" in capsys.readouterr().err
