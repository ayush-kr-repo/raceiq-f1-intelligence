import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import MODELS_DIR, OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR, RUNS_DIR, VERIFY_DIR


REQUIRED_MODULES = [
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "plotly",
    "streamlit",
]

OPTIONAL_MODULES = [
    "xgboost",
    "optuna",
    "shap",
    "fastf1",
]


def check_modules(names):
    results = {}
    for name in names:
        try:
            importlib.import_module(name)
            results[name] = "ok"
        except Exception as exc:
            results[name] = f"missing: {exc}"
    return results


def main():
    report = {
        "project_root": str(ROOT),
        "required_modules": check_modules(REQUIRED_MODULES),
        "optional_modules": check_modules(OPTIONAL_MODULES),
        "required_files": {
            "run.py": (ROOT / "run.py").exists(),
            "app.py": (ROOT / "app.py").exists(),
            "data/raw_data.csv": (ROOT / "data" / "raw_data.csv").exists(),
        },
        "required_dirs": {
            "outputs": OUTPUT_DIR.exists(),
            "models": MODELS_DIR.exists(),
            "plots": PLOTS_DIR.exists(),
            "reports": REPORTS_DIR.exists(),
            "runs": RUNS_DIR.exists(),
        },
    }
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    verify_path = VERIFY_DIR / "setup_report.json"
    verify_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
