from pathlib import Path
import json
import sys

import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app


def test_get_latest_run_manifest_uses_latest_file(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    latest_dir = runs_dir / "run_20260423_220000"
    older_dir = runs_dir / "run_20260422_210000"
    latest_dir.mkdir(parents=True, exist_ok=True)
    older_dir.mkdir(parents=True, exist_ok=True)

    (older_dir / "manifest.json").write_text(json.dumps({"run_id": "old"}), encoding="utf-8")
    (latest_dir / "manifest.json").write_text(json.dumps({"run_id": "latest"}), encoding="utf-8")

    monkeypatch.setattr(app, "RUNS_DIR", runs_dir)
    manifest = app.get_latest_run_manifest()

    assert manifest["run_id"] == "latest"
