from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.preprocessor as preprocessor_module
import src.trainer as trainer_module
from src.preprocessor import F1Preprocessor
from src.trainer import F1ModelTrainer


def test_trainer_returns_expected_registry(tmp_path, synthetic_csv, monkeypatch):
    out_dir = tmp_path / "outputs"
    reports_dir = out_dir / "reports"
    models_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(preprocessor_module, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(preprocessor_module, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(trainer_module, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(trainer_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(trainer_module, "REPORTS_DIR", reports_dir)

    def fake_xgb(self, X_train, y_train, X_val, y_val):
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        return model

    monkeypatch.setattr(F1ModelTrainer, "_train_xgb", fake_xgb)

    splits = F1Preprocessor().fit_transform(str(synthetic_csv))
    registry = F1ModelTrainer().train(splits)

    assert {"lr", "rf", "xgb", "ensemble", "finisher_regressor", "two_stage", "rankblend"}.issubset(
        set(registry["regressors"].keys())
    )
    assert {"dnf", "podium", "points"} == set(registry["classifiers"].keys())
    assert (models_dir / "two_stage_model.pkl").exists()
