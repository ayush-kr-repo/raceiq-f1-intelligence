from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.evaluator as evaluator_module
import src.preprocessor as preprocessor_module
from src.evaluator import F1Evaluator
from src.preprocessor import F1Preprocessor


class FakeRegressor:
    def __init__(self, offset=0.0):
        self.offset = offset

    def predict(self, X):
        base = np.linspace(1, min(21, X.shape[0]), X.shape[0])
        return base + self.offset


class FakeClassifier:
    def predict_proba(self, X):
        probs = np.full((X.shape[0], 2), 0.5, dtype=float)
        probs[:, 0] = 1.0 - probs[:, 1]
        return probs


def test_evaluator_writes_key_artifacts(tmp_path, synthetic_csv, monkeypatch):
    out_dir = tmp_path / "outputs"
    reports_dir = out_dir / "reports"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(preprocessor_module, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(preprocessor_module, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(evaluator_module, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(evaluator_module, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(evaluator_module, "PLOTS_DIR", plots_dir)

    splits = F1Preprocessor().fit_transform(str(synthetic_csv))
    registry = {
        "regressors": {
            "lr": FakeRegressor(0.0),
            "rf": FakeRegressor(0.2),
            "ensemble": FakeRegressor(0.1),
            "two_stage": FakeRegressor(0.3),
            "rankblend": FakeRegressor(0.15),
        },
        "classifiers": {
            "dnf": FakeClassifier(),
            "podium": FakeClassifier(),
            "points": FakeClassifier(),
        },
    }

    report = F1Evaluator().evaluate(registry, splits)

    assert "two_stage" in report
    assert (reports_dir / "test_predictions.csv").exists()
    assert (reports_dir / "metrics_report.json").exists()
    assert (plots_dir / "model_comparison.png").exists()
