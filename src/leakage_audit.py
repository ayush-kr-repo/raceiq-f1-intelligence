import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import REPORTS_DIR


class LeakageAuditor:
    """Run lightweight leakage checks over historical feature columns."""

    def audit(self, df: pd.DataFrame) -> dict:
        ordered = df.sort_values(["year", "round", "race_id", "driver"], kind="stable").reset_index(drop=True)

        driver_seen = set()
        team_seen = set()
        driver_circuit_seen = set()

        violations = {
            "driver_recent_form_without_history": 0,
            "constructor_recent_form_without_history": 0,
            "track_specific_score_without_history": 0,
        }

        for _, row in ordered.iterrows():
            driver_key = row["driver"]
            team_key = row["team"]
            driver_circuit_key = (row["driver"], row["circuit"])

            if driver_key not in driver_seen and pd.notna(row.get("driver_recent_form_score")):
                violations["driver_recent_form_without_history"] += 1
            if team_key not in team_seen and pd.notna(row.get("constructor_recent_form_score")):
                violations["constructor_recent_form_without_history"] += 1
            if driver_circuit_key not in driver_circuit_seen and pd.notna(row.get("track_specific_driver_score")):
                violations["track_specific_score_without_history"] += 1

            driver_seen.add(driver_key)
            team_seen.add(team_key)
            driver_circuit_seen.add(driver_circuit_key)

        report = {
            "status": "passed" if sum(violations.values()) == 0 else "warning",
            "rows_audited": int(len(ordered)),
            "violations": violations,
            "notes": [
                "Rolling and historical features are expected to be null until prior history exists.",
                "Explainer telemetry outputs are post-race artifacts and should not be used as predictor inputs.",
            ],
        }

        output_path = REPORTS_DIR / "leakage_audit.json"
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
