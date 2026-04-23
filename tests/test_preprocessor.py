from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.preprocessor as preprocessor_module
from src.preprocessor import F1Preprocessor


def test_preprocessor_builds_splits_and_audit(tmp_path, synthetic_csv, monkeypatch):
    out_dir = tmp_path / "outputs"
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(preprocessor_module, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(preprocessor_module, "REPORTS_DIR", reports_dir)

    preprocessor = F1Preprocessor()
    splits = preprocessor.fit_transform(str(synthetic_csv))

    assert splits["X_train"].shape[0] > 0
    assert splits["X_val"].shape[0] > 0
    assert splits["X_test"].shape[0] > 0
    assert len(splits["feature_names"]) > 0
    assert splits["leakage_audit"]["status"] in {"passed", "warning"}
