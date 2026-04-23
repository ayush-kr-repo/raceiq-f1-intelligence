from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.ablation as ablation_module
import src.preprocessor as preprocessor_module
from src.ablation import FeatureAblationRunner
from src.preprocessor import F1Preprocessor


def test_ablation_runner_writes_reports(tmp_path, synthetic_csv, monkeypatch):
    out_dir = tmp_path / "outputs"
    reports_dir = out_dir / "reports"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(preprocessor_module, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(preprocessor_module, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(ablation_module, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(ablation_module, "PLOTS_DIR", plots_dir)

    splits = F1Preprocessor().fit_transform(str(synthetic_csv))
    final_report = {
        "ensemble": {"mae": 2.0, "rmse": 2.5, "spearman_r": 0.4},
        "two_stage": {"mae": 1.8, "rmse": 2.3, "spearman_r": 0.48},
        "rankblend": {"mae": 1.9, "rmse": 2.4, "spearman_r": 0.45},
    }

    report_df = FeatureAblationRunner().run(splits, final_report)

    assert not report_df.empty
    assert (reports_dir / "ablation_report.csv").exists()
    assert (plots_dir / "ablation_study.png").exists()
