from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.leakage_audit as leakage_module
from src.feature_builder import F1FeatureBuilder
from src.leakage_audit import LeakageAuditor


def test_leakage_audit_writes_report(tmp_path, synthetic_raw_df, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(leakage_module, "REPORTS_DIR", reports_dir)

    enriched = F1FeatureBuilder().build(synthetic_raw_df)
    report = LeakageAuditor().audit(enriched)

    assert report["status"] in {"passed", "warning"}
    assert "violations" in report
    assert (reports_dir / "leakage_audit.json").exists()
