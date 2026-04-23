import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DNF_HISTORY_WINDOW,
    DNF_SENTINEL,
    HIGH_DOWNFORCE_CIRCUITS,
    OVERTAKE_DIFFICULTY,
    POWER_CIRCUITS,
    RECENT_FORM_WINDOW,
    STREET_CIRCUITS,
    TEAM_FORM_WINDOW,
    TRACK_HISTORY_WINDOW,
)

log = logging.getLogger(__name__)


class F1FeatureBuilder:
    """Enrich the flat race dataset with historical and contextual features."""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame = frame.sort_values(["year", "round", "race_id", "driver"], kind="stable").reset_index(drop=True)

        frame["compound_start"] = frame.get("compound_start", pd.Series(index=frame.index)).fillna("UNKNOWN")
        frame["circuit_type"] = frame["circuit"].map(self._infer_circuit_type)
        frame["is_street_circuit"] = frame["circuit"].isin(STREET_CIRCUITS).astype(int)
        frame["season_phase"] = frame["round"].apply(self._season_phase)
        frame["pit_stop_count"] = frame.get("pit_stop_count", frame.get("stint_count", 1).fillna(1) - 1).clip(lower=0)
        frame["strategy_delta"] = np.where(
            frame["finish_position"].lt(DNF_SENTINEL),
            frame["grid_position"].fillna(0) - frame["finish_position"].fillna(DNF_SENTINEL),
            -3.0,
        )

        if "avg_speed_trap" not in frame.columns:
            frame["avg_speed_trap"] = np.nan
        if "speed_trap_percentile" not in frame.columns:
            frame["speed_trap_percentile"] = np.nan
        if "chaos_index" not in frame.columns:
            race_chaos = frame.groupby("race_id").apply(
                lambda group: float(group["is_dnf"].mean()) + 0.5 * float(group["is_wet"].max())
            )
            frame["chaos_index"] = frame["race_id"].map(race_chaos.to_dict())

        enriched_rows = []
        for idx, row in frame.iterrows():
            past = frame.iloc[:idx]
            driver_past = past[past["driver"] == row["driver"]]
            team_past = past[past["team"] == row["team"]]
            same_circuit_past = driver_past[driver_past["circuit"] == row["circuit"]]
            wet_past = driver_past[driver_past["is_wet"] == 1]
            street_past = driver_past[driver_past["is_street_circuit"] == 1]
            chaos_past = driver_past[driver_past["chaos_index"] >= 0.35]

            enriched_rows.append(
                {
                    "driver_recent_form_score": self._weighted_finish_score(driver_past.tail(RECENT_FORM_WINDOW)),
                    "constructor_recent_form_score": self._weighted_points_score(team_past.tail(TEAM_FORM_WINDOW)),
                    "track_specific_driver_score": self._weighted_finish_score(same_circuit_past.tail(TRACK_HISTORY_WINDOW)),
                    "wet_performance_score": self._weighted_finish_score(wet_past.tail(RECENT_FORM_WINDOW)),
                    "street_circuit_score": self._weighted_finish_score(street_past.tail(RECENT_FORM_WINDOW)),
                    "overtake_difficulty_factor": self._overtake_factor(row["circuit"]),
                    "pit_stop_efficiency": self._weighted_mean(self._pit_efficiency_series(team_past.tail(TEAM_FORM_WINDOW))),
                    "safety_car_sensitivity": self._weighted_mean(
                        np.abs(chaos_past.tail(RECENT_FORM_WINDOW)["strategy_delta"].to_numpy())
                    ),
                    "quali_to_race_conversion": self._weighted_mean(
                        team_past.tail(TEAM_FORM_WINDOW)["strategy_delta"].to_numpy()
                    ),
                    "dnf_risk_score": self._dnf_risk(driver_past.tail(DNF_HISTORY_WINDOW), team_past.tail(DNF_HISTORY_WINDOW)),
                    "avg_top_speed_last_5": self._weighted_mean(driver_past.tail(RECENT_FORM_WINDOW)["avg_speed_trap"].to_numpy()),
                    "straight_line_speed_percentile_last_5": self._weighted_mean(
                        driver_past.tail(RECENT_FORM_WINDOW)["speed_trap_percentile"].to_numpy()
                    ),
                    "avg_deg_same_compound_last_5": self._weighted_mean(
                        driver_past[driver_past["compound_start"] == row["compound_start"]]
                        .tail(RECENT_FORM_WINDOW)["tyre_deg_slope"]
                        .to_numpy()
                    ),
                    "constructor_strategy_score": self._weighted_mean(
                        self._strategy_score_series(team_past.tail(TEAM_FORM_WINDOW))
                    ),
                }
            )

        enriched = pd.concat([frame, pd.DataFrame(enriched_rows)], axis=1)
        race_strength = enriched.groupby("race_id")["driver_recent_form_score"].transform("mean")
        enriched["field_strength"] = race_strength.fillna(race_strength.mean())
        return enriched

    def _infer_circuit_type(self, circuit: str) -> str:
        if circuit in STREET_CIRCUITS:
            return "street"
        if circuit in POWER_CIRCUITS:
            return "power"
        if circuit in HIGH_DOWNFORCE_CIRCUITS:
            return "high_downforce"
        return "balanced"

    def _season_phase(self, round_number: float) -> str:
        if pd.isna(round_number):
            return "mid"
        if round_number <= 7:
            return "early"
        if round_number <= 16:
            return "mid"
        return "late"

    def _overtake_factor(self, circuit: str) -> float:
        return OVERTAKE_DIFFICULTY.get(circuit, 0.5)

    def _weighted_finish_score(self, subset: pd.DataFrame) -> float:
        if subset.empty:
            return np.nan
        positions = subset["finish_position"].replace(DNF_SENTINEL, DNF_SENTINEL + 1).astype(float).to_numpy()
        scores = np.clip((DNF_SENTINEL + 1 - positions) / DNF_SENTINEL, 0.0, 1.0)
        return self._weighted_mean(scores)

    def _weighted_points_score(self, subset: pd.DataFrame) -> float:
        if subset.empty:
            return np.nan
        points = np.array([self._finish_to_points(pos) for pos in subset["finish_position"].astype(float).to_numpy()])
        return self._weighted_mean(points / 25.0)

    def _finish_to_points(self, position: float) -> float:
        table = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        if math.isnan(position):
            return 0.0
        return float(table.get(int(position), 0.0))

    def _pit_efficiency_series(self, subset: pd.DataFrame) -> np.ndarray:
        if subset.empty:
            return np.array([], dtype=float)
        efficiency = subset["strategy_delta"].to_numpy(dtype=float) / np.maximum(subset["pit_stop_count"].to_numpy(dtype=float), 1.0)
        return efficiency

    def _strategy_score_series(self, subset: pd.DataFrame) -> np.ndarray:
        if subset.empty:
            return np.array([], dtype=float)
        return (
            subset["strategy_delta"].to_numpy(dtype=float)
            - 0.35 * subset["pit_stop_count"].to_numpy(dtype=float)
        )

    def _dnf_risk(self, driver_past: pd.DataFrame, team_past: pd.DataFrame) -> float:
        driver_rate = self._weighted_mean(driver_past["is_dnf"].to_numpy(dtype=float))
        team_rate = self._weighted_mean(team_past["is_dnf"].to_numpy(dtype=float))
        if np.isnan(driver_rate) and np.isnan(team_rate):
            return np.nan
        if np.isnan(driver_rate):
            return team_rate
        if np.isnan(team_rate):
            return driver_rate
        return round(0.7 * driver_rate + 0.3 * team_rate, 4)

    def _weighted_mean(self, values) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.nan
        weights = np.arange(1, arr.size + 1, dtype=float)
        return float(np.average(arr, weights=weights))
