import sys
import unittest
from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ProjectSmokeTests(unittest.TestCase):
    def test_core_python_files_compile(self):
        python_files = [
            ROOT / "config.py",
            ROOT / "run.py",
            ROOT / "app.py",
            ROOT / "src" / "collector.py",
            ROOT / "src" / "evaluator.py",
            ROOT / "src" / "explainer.py",
            ROOT / "src" / "ablation.py",
            ROOT / "src" / "experiment_tracker.py",
            ROOT / "src" / "feature_builder.py",
            ROOT / "src" / "leakage_audit.py",
            ROOT / "src" / "preprocessor.py",
            ROOT / "src" / "trainer.py",
        ]
        for path in python_files:
            py_compile.compile(str(path), doraise=True)

    def test_expected_files_exist(self):
        expected = [
            ROOT / "run.py",
            ROOT / "app.py",
            ROOT / "config.py",
            ROOT / "requirements.txt",
            ROOT / "src" / "collector.py",
            ROOT / "src" / "evaluator.py",
            ROOT / "src" / "explainer.py",
            ROOT / "src" / "ablation.py",
            ROOT / "src" / "experiment_tracker.py",
            ROOT / "src" / "feature_builder.py",
            ROOT / "src" / "leakage_audit.py",
            ROOT / "src" / "preprocessor.py",
            ROOT / "src" / "trainer.py",
            ROOT / "scripts" / "verify_setup.py",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"Missing expected file: {path}")

    def test_raw_data_exists(self):
        self.assertTrue((ROOT / "data" / "raw_data.csv").exists())


if __name__ == "__main__":
    unittest.main()
