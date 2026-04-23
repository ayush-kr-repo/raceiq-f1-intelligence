import json
import subprocess
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    """Persist lightweight run manifests and artifacts for reproducibility."""

    def __init__(self, runs_dir: Path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        self.run_dir = runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "run_id": self.run_id,
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "git_hash": self._git_hash(),
            "params": {},
            "metrics": {},
            "artifacts": [],
        }

    def log_params(self, payload: dict):
        self.manifest["params"].update(payload)
        self._flush()

    def log_metrics(self, payload: dict):
        self.manifest["metrics"].update(payload)
        self._flush()

    def log_artifact(self, path: str):
        self.manifest["artifacts"].append(path)
        self._flush()

    def complete(self, status: str = "completed"):
        self.manifest["status"] = status
        self.manifest["finished_at"] = datetime.now().isoformat()
        started_at = datetime.fromisoformat(self.manifest["started_at"])
        finished_at = datetime.fromisoformat(self.manifest["finished_at"])
        self.manifest["runtime_seconds"] = round((finished_at - started_at).total_seconds(), 3)
        self._flush()

    def _flush(self):
        (self.run_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    def _git_hash(self):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                check=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return "unavailable"
