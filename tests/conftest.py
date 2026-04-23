from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_raw_df():
    drivers = ["VER", "NOR", "LEC", "HAM"]
    teams = {"VER": "Red Bull", "NOR": "McLaren", "LEC": "Ferrari", "HAM": "Mercedes"}
    circuits = ["Bahrain", "Monaco", "Silverstone", "Monza", "Singapore", "Austin", "Suzuka"]
    compounds = ["SOFT", "MEDIUM", "HARD", "SOFT"]

    rows = []
    for year in range(2018, 2025):
        for round_no in range(1, 5):
            for idx, driver in enumerate(drivers):
                finish_position = ((idx + round_no + year) % len(drivers)) + 1
                is_dnf = 1 if (round_no == 3 and driver in {"HAM", "LEC"} and year >= 2021) else 0
                if is_dnf:
                    finish_position = 21
                rows.append(
                    {
                        "race_id": f"{year}_{round_no:02d}",
                        "year": year,
                        "round": round_no,
                        "gp_name": f"GP {round_no}",
                        "circuit": circuits[(round_no + idx) % len(circuits)],
                        "driver": driver,
                        "team": teams[driver],
                        "finish_position": finish_position,
                        "is_dnf": is_dnf,
                        "grid_position": idx + 1,
                        "quali_gap_to_pole": round(0.01 * idx + 0.001 * round_no, 4),
                        "lap_consistency_std": round(0.12 + 0.03 * idx, 4),
                        "tyre_deg_slope": round(0.015 + 0.004 * idx, 4),
                        "stint_count": 2 + (round_no % 2),
                        "pit_stop_count": 1 + (round_no % 2),
                        "compound_start": compounds[idx],
                        "air_temp": 24 + round_no,
                        "track_temp": 33 + round_no,
                        "temp_delta": 9,
                        "is_wet": 1 if round_no == 2 else 0,
                        "humidity": 48 + idx,
                        "avg_speed_trap": 308 + idx * 3 + round_no,
                        "speed_trap_percentile": 0.55 + idx * 0.08,
                        "strategy_delta": (idx + 1) - finish_position if finish_position < 21 else -3,
                        "chaos_index": 0.2 + 0.1 * (round_no % 2),
                        "driver_recent_avg_pos": None,
                        "constructor_dnf_rate": None,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_csv(tmp_path, synthetic_raw_df):
    path = tmp_path / "synthetic_raw.csv"
    synthetic_raw_df.to_csv(path, index=False)
    return path
