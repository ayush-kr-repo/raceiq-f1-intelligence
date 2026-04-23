from pathlib import Path
import sys

import pandas as pd
import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app


def test_dashboard_helpers_load_files(tmp_path):
    json_path = tmp_path / "sample.json"
    csv_path = tmp_path / "sample.csv"

    json_path.write_text('{"ok": true}', encoding="utf-8")
    pd.DataFrame({"driver": ["VER"], "score": [0.9]}).to_csv(csv_path, index=False)

    assert app.load_json(json_path, {})["ok"] is True
    df = app.load_dataframe(csv_path)
    assert list(df.columns) == ["driver", "score"]
