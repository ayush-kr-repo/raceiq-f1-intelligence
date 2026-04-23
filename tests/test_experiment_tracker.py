from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_tracker import ExperimentTracker


def test_experiment_tracker_writes_manifest(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    tracker.log_params({"feature_count": 12})
    tracker.log_metrics({"xgb_mae": 2.1})
    tracker.log_artifact("outputs/reports/test_predictions.csv")
    tracker.complete()

    manifest_path = tracker.run_dir / "manifest.json"
    assert manifest_path.exists()
    content = manifest_path.read_text(encoding="utf-8")
    assert '"runtime_seconds"' in content
    assert '"feature_count"' in content
