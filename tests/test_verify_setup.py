from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_verify_setup_script_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_setup.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"required_modules"' in result.stdout
    assert (ROOT / "outputs" / "verify" / "setup_report.json").exists()
