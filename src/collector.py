# =============================================================================
# src/collector.py — data collection + feature extraction
#
# What this file does:
#   1. Downloads race + qualifying sessions from the official F1 API via FastF1
#   2. Computes all engineered features (consistency, tyre deg, quali gap, etc.)
#   3. Returns one flat DataFrame where each row = one driver in one race
#
# FastF1 overview:
#   FastF1 is a Python library that wraps the official F1 timing API.
#   It provides lap-by-lap timing, tyre data, weather, and race results.
#   Calls are cached locally so repeat runs are fast (first run is slow).
#
# Data shape you'll end up with:
#   ~20 drivers × ~22 races × 7 seasons ≈ 3,080 rows
#   Each row has ~20 feature columns + 1 target column (finish_position)
# =============================================================================

import fastf1
from fastf1.core import DataNotLoadedError
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import CACHE_DIR, DNF_SENTINEL, MIN_STINT_LAPS, N_CLEAN_LAPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Main collector class
# =============================================================================

class F1DataCollector:
    """
    Collects F1 race data and computes features for all seasons.

    Usage:
        collector = F1DataCollector()
        df = collector.collect(seasons=[2021, 2022, 2023])
        df.to_csv("data/raw_data.csv", index=False)
    """

    def __init__(self):
        # Enable FastF1 disk cache. Without this, every run re-downloads
        # gigabytes of timing data. The cache stores API responses as files.
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        log.info(f"FastF1 cache: {CACHE_DIR}")

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def collect(self, seasons: list[int]) -> pd.DataFrame:
        """
        Loops over all seasons and races, returns one flat DataFrame.

        Parameters:
            seasons: list of years, e.g. [2018, 2019, ..., 2024]

        Returns:
            DataFrame with one row per driver per race
        """
        all_rows = []

        for year in seasons:
            log.info(f"--- Season {year} ---")
            schedule = self._get_schedule(year)

            for _, event in schedule.iterrows():
                rows = self._process_race(year, event)
                if rows:
                    all_rows.extend(rows)

        if not all_rows:
            raise RuntimeError("No data collected. Is FastF1 installed correctly?")

        df = pd.DataFrame(all_rows)
        log.info(f"Collected {len(df)} rows across {df['race_id'].nunique()} races")

        # Add historical features that need the full dataset to compute
        df = self._add_historical_features(df)
        return df

    # -------------------------------------------------------------------------
    # Season schedule
    # -------------------------------------------------------------------------

    def _get_schedule(self, year: int) -> pd.DataFrame:
        """
        Returns list of Grand Prix events for a season.
        Filters out pre-season testing which has no race results.
        """
        return fastf1.get_event_schedule(year, include_testing=False)

    # -------------------------------------------------------------------------
    # Per-race processing
    # -------------------------------------------------------------------------

    def _process_race(self, year: int, event) -> list[dict]:
        """
        Processes one Grand Prix. Returns a list of dicts (one per driver).

        Steps:
          1. Load race session (laps, results, weather)
          2. Load qualifying session (best lap per driver)
          3. For each driver, build a feature dictionary
        """
        gp_name  = event["EventName"]
        round_no = event["RoundNumber"]
        race_id  = f"{year}_{round_no:02d}"

        try:
            # Load race session
            # telemetry=False  → skip telemetry (not needed, saves ~80% of download time)
            # weather=True     → load weather data (we need temps)
            # messages=False   → skip team radio (not needed)
            race = fastf1.get_session(year, round_no, "R")
            race.load(telemetry=False, weather=True, messages=False)

            # Load qualifying session (for quali gap feature)
            quali = fastf1.get_session(year, round_no, "Q")
            quali.load(telemetry=False, weather=False, messages=False)

        except Exception as e:
            log.warning(f"Skipping {gp_name} {year}: {e}")
            return []
        try:
            laps    = race.laps
            results = race.results
        except DataNotLoadedError as e:
            log.warning(f"Skipping {gp_name} {year} : race data not fully loaded ({e})")
            return []

        if laps is None or len(laps) == 0 or results is None or len(results) == 0:
            log.warning(f"Empty session data for {gp_name} {year}")
            return []

        # Pre-compute weather once (same for all drivers in the race)
        weather_feats = self._extract_weather(race)

        # Pre-compute qualifying gaps once (same session, different driver times)
        quali_gaps = self._compute_all_quali_gaps(quali)

        # Starting compound (from first lap of each driver)
        starting_compounds = self._get_starting_compounds(laps)
        speed_trap_metrics = self._compute_speed_trap_metrics(laps)
        race_chaos_index = self._compute_race_chaos_index(results, weather_feats)

        rows = []
        for _, result_row in results.iterrows():
            driver = result_row.get("Abbreviation", None)
            if driver is None:
                continue

            driver_laps = laps.pick_driver(driver)
            row = self._build_row(
                driver=driver,
                result=result_row,
                driver_laps=driver_laps,
                quali_gaps=quali_gaps,
                starting_compounds=starting_compounds,
                weather_feats=weather_feats,
                speed_trap_metrics=speed_trap_metrics,
                race_chaos_index=race_chaos_index,
                race_id=race_id,
                year=year,
                gp_name=gp_name,
                circuit=event.get("Location", gp_name),
                round_no=round_no,
            )
            if row:
                rows.append(row)

        return rows

    # -------------------------------------------------------------------------
    # Build one driver's feature dict
    # -------------------------------------------------------------------------

    def _build_row(self, driver, result, driver_laps, quali_gaps,
                   starting_compounds, weather_feats, speed_trap_metrics,
                   race_chaos_index, race_id,
                   year, gp_name, circuit, round_no) -> dict | None:
        """
        Assembles the full feature dict for one driver in one race.
        This is where all the features we designed land in one place.
        """
        row = {
            # Identifiers (not used as features — used to join/filter later)
            "race_id":    race_id,
            "year":       year,
            "round":      round_no,
            "gp_name":    gp_name,
            "circuit":    circuit,
            "driver":     driver,
            "team":       result.get("TeamName", "Unknown"),
        }

        # -----------------------------------------------------------------
        # TARGET VARIABLE: finish_position
        # -----------------------------------------------------------------
        # ClassifiedPosition is the official FIA classification.
        # Status tells us HOW the race ended (Finished, Retired, Disqualified…)
        #
        # DNF strategy: assign sentinel=21 and set is_dnf=1.
        # Do NOT drop DNF rows. They carry real information — the model must
        # learn that certain drivers/teams have higher retirement risk.
        #
        status      = str(result.get("Status", ""))
        classified  = result.get("ClassifiedPosition", None)
        finished    = any(status.startswith(s) for s in ["Finished", "+"])

        try:
            position = int(str(classified))
        except (ValueError, TypeError):
            position = DNF_SENTINEL   # DNF, DSQ, NC → sentinel 21

        row["finish_position"] = position
        row["is_dnf"]          = 0 if finished else 1

        # -----------------------------------------------------------------
        # GRID POSITION (from qualifying result)
        # -----------------------------------------------------------------
        # Grid position = where you start on the grid.
        # It's the single strongest predictor — overtaking is hard in F1.
        try:
            row["grid_position"] = int(result.get("GridPosition", 0))
        except (ValueError, TypeError):
            row["grid_position"] = 0

        # -----------------------------------------------------------------
        # QUALIFYING GAP TO POLE (relative, circuit-normalised)
        # -----------------------------------------------------------------
        # Formula: (driver_time - pole_time) / pole_time
        # Pre-computed for the whole session; just look up this driver.
        row["quali_gap_to_pole"] = quali_gaps.get(driver, np.nan)

        # -----------------------------------------------------------------
        # LAP TIME CONSISTENCY
        # -----------------------------------------------------------------
        # Std deviation of the last N clean lap times.
        # Lower = more consistent driver = better race management.
        consistency = self._compute_lap_consistency(driver_laps)
        row["lap_consistency_std"] = consistency["std"]
        row["lap_consistency_n"]   = consistency["n"]   # diagnostic column

        # -----------------------------------------------------------------
        # TYRE DEGRADATION SLOPE
        # -----------------------------------------------------------------
        # Slope (seconds/lap) from linear regression of LapTime ~ LapNumber
        # per stint. Higher slope = tyres degrading faster = worse management.
        row["tyre_deg_slope"] = self._compute_tyre_degradation(driver_laps)

        # -----------------------------------------------------------------
        # STINT COUNT
        # -----------------------------------------------------------------
        # = number of pit stops + 1
        # A driver with 3 stints made 2 pit stops.
        stint_count = 1
        if len(driver_laps) > 0 and "Stint" in driver_laps.columns:
            max_stint = driver_laps["Stint"].dropna().max()
            if pd.notna(max_stint):
                stint_count = int(max_stint)
        row["stint_count"] = stint_count
        row["pit_stop_count"] = max(stint_count - 1, 0)


        # -----------------------------------------------------------------
        # STARTING TYRE COMPOUND
        # -----------------------------------------------------------------
        row["compound_start"] = starting_compounds.get(driver, "UNKNOWN")
        row["avg_speed_trap"] = speed_trap_metrics.get(driver, {}).get("avg_speed_trap", np.nan)
        row["speed_trap_percentile"] = speed_trap_metrics.get(driver, {}).get("speed_trap_percentile", np.nan)
        row["strategy_delta"] = (
            row["grid_position"] - row["finish_position"]
            if row["finish_position"] < DNF_SENTINEL
            else -3
        )
        row["chaos_index"] = race_chaos_index

        # -----------------------------------------------------------------
        # WEATHER FEATURES (same for all drivers in this race)
        # -----------------------------------------------------------------
        row.update(weather_feats)

        # Historical features will be added later (require full dataset)
        row["driver_recent_avg_pos"] = np.nan
        row["constructor_dnf_rate"]  = np.nan

        return row

    # =========================================================================
    # Feature computation methods
    # =========================================================================

    def _compute_all_quali_gaps(self, quali_session) -> dict:
        """
        Computes qualifying gap to pole for ALL drivers in one pass.

        Returns: {driver_abbreviation: gap_float}

        Why normalise?
            Monaco pole ≈ 72s, Monza pole ≈ 80s.
            An absolute gap of 0.5s means very different things.
            Relative gap is circuit-independent and comparable.

        Formula: (driver_best - pole_best) / pole_best
        Pole always gets gap = 0.0; slower drivers get positive values.
        """
        gaps = {}
        try:
            try: 
                q_laps = quali_session.laps
            except DataNotLoadedError:
                return gaps

            if q_laps is None or len(q_laps) == 0:
                return gaps

            # Best lap per driver (minimum LapTime across all Q1/Q2/Q3 attempts)
            best = (
                q_laps[q_laps["LapTime"].notna()]
                .groupby("Driver")["LapTime"]
                .min()
            )

            if best.empty:
                return gaps

            pole_time = best.min().total_seconds()
            if pole_time <= 0:
                return gaps

            for driver, t in best.items():
                driver_time = t.total_seconds()
                gaps[driver] = round((driver_time - pole_time) / pole_time, 6)

        except Exception as e:
            log.warning(f"Quali gap error: {e}")

        return gaps

    def _compute_speed_trap_metrics(self, laps: pd.DataFrame) -> dict:
        metrics = {}
        try:
            if "SpeedST" not in laps.columns:
                return metrics
            valid = laps[laps["SpeedST"].notna()].copy()
            if valid.empty:
                return metrics

            driver_speed = valid.groupby("Driver")["SpeedST"].median()
            percentiles = driver_speed.rank(pct=True)
            for driver in driver_speed.index:
                metrics[driver] = {
                    "avg_speed_trap": round(float(driver_speed[driver]), 3),
                    "speed_trap_percentile": round(float(percentiles[driver]), 4),
                }
        except Exception as exc:
            log.debug(f"Speed trap metric error: {exc}")
        return metrics

    def _compute_race_chaos_index(self, results: pd.DataFrame, weather_feats: dict) -> float:
        try:
            dnf_rate = float(results["Status"].astype(str).str.startswith(("Finished", "+")).map(lambda finished: 0 if finished else 1).mean())
            wet_bonus = 0.5 * float(weather_feats.get("is_wet", 0))
            return round(dnf_rate + wet_bonus, 4)
        except Exception:
            return float(weather_feats.get("is_wet", 0)) * 0.5

    def _compute_lap_consistency(self, driver_laps, n: int = N_CLEAN_LAPS) -> dict:
        """
        Computes lap time consistency: std dev of last N clean laps.

        'Clean lap' definition:
          1. TrackStatus == '1'  →  green flag, no SC or VSC
          2. PitInTime is NaT   →  not pitting in this lap
          3. PitOutTime is NaT  →  not pitting out this lap
          4. LapTime is valid   →  not NaN

        We take LAST N laps because end-of-race pace is more representative:
          - Traffic has spread out
          - Tyre behaviour is stable (not first lap on fresh rubber)

        We cap outlier laps at 1.5× median:
          - Catches SC laps that FastF1 mislabelled as green
          - Catches laps ruined by track limits investigations

        Returns: {"std": float, "n": int}
        """
        try:
            laps = driver_laps.copy()
            laps = laps[laps["LapTime"].notna()].copy()

            if len(laps) == 0:
                return {"std": np.nan, "n": 0}

            # Convert timedelta to float seconds
            laps["lap_s"] = laps["LapTime"].dt.total_seconds()

            # Filter: green flag only
            laps = laps[laps["TrackStatus"] == "1"]

            # Filter: no pit laps
            laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]

            # Filter: remove outlier lap times (SC laps sometimes mislabelled)
            if len(laps) > 2:
                med = laps["lap_s"].median()
                laps = laps[laps["lap_s"] < med * 1.5]

            if len(laps) < 3:
                return {"std": np.nan, "n": len(laps)}

            # Take last N (sort descending by LapNumber, take first n rows)
            last_n = laps.nlargest(min(n, len(laps)), "LapNumber")
            return {"std": round(float(last_n["lap_s"].std()), 4), "n": len(laps)}

        except Exception as e:
            log.debug(f"Lap consistency error: {e}")
            return {"std": np.nan, "n": 0}

    def _compute_tyre_degradation(self, driver_laps, min_laps: int = MIN_STINT_LAPS) -> float:
        """
        Computes average tyre degradation slope across all stints.

        For each stint, we fit:
            LapTime (seconds) = m * LapNumber + c

        The coefficient m is the degradation slope:
          - m > 0 → tyres are getting slower each lap (expected, positive)
          - m near 0 → very strong tyre management
          - m large → rapid degradation

        We only use clean laps (same filters as consistency above).
        We require MIN_STINT_LAPS per stint for a reliable fit.
        We average slopes across all valid stints.

        Why linear regression instead of raw delta?
            A single fast/slow lap ruins a delta calculation.
            Linear regression smooths over the whole stint — much more stable.
        """
        try:
            laps = driver_laps.copy()
            laps = laps[laps["LapTime"].notna()].copy()
            laps["lap_s"] = laps["LapTime"].dt.total_seconds()

            # Same clean lap filters as consistency
            laps = laps[laps["TrackStatus"] == "1"]
            laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]

            if len(laps) > 2:
                med = laps["lap_s"].median()
                laps = laps[laps["lap_s"] < med * 1.5]

            slopes = []
            for stint_id in laps["Stint"].unique():
                stint = laps[laps["Stint"] == stint_id].copy()

                if len(stint) < min_laps:
                    continue  # too few laps → unreliable slope

                # X: lap number (reshaped for sklearn)
                X = stint["LapNumber"].values.reshape(-1, 1)
                y = stint["lap_s"].values

                model = LinearRegression()
                model.fit(X, y)
                slopes.append(model.coef_[0])   # seconds per lap

            return round(float(np.mean(slopes)), 6) if slopes else np.nan

        except Exception as e:
            log.debug(f"Tyre deg error: {e}")
            return np.nan

    def _extract_weather(self, race_session) -> dict:
        """
        Extracts race-wide weather features.

        We take MEDIAN values (robust to brief anomalies like a
        sudden rain shower that ends quickly).

        Features:
          air_temp   → ambient temperature (°C)
          track_temp → asphalt surface temperature (°C)
          temp_delta → track_temp - air_temp (how much sun the track absorbs)
          is_wet     → 1 if any rainfall occurred during the race
          humidity   → relative humidity (%)
        """
        try:
            try:
                w = race_session.weather_data
            except DataNotLoadedError:
                return self._empty_weather()
            
            if w is None or len(w) == 0:
                return self._empty_weather()

            air   = float(w["AirTemp"].median())
            track = float(w["TrackTemp"].median())
            wet   = bool(w["Rainfall"].any()) if "Rainfall" in w.columns else False
            hum   = float(w["Humidity"].median()) if "Humidity" in w.columns else np.nan

            return {
                "air_temp":   round(air, 2),
                "track_temp": round(track, 2),
                "temp_delta": round(track - air, 2),
                "is_wet":     int(wet),
                "humidity":   round(hum, 2) if not np.isnan(hum) else np.nan,
            }
        except Exception:
            return self._empty_weather()

    def _empty_weather(self) -> dict:
        return {"air_temp": np.nan, "track_temp": np.nan,
                "temp_delta": np.nan, "is_wet": 0, "humidity": np.nan}

    def _get_starting_compounds(self, laps: pd.DataFrame) -> dict:
        """
        Returns the starting tyre compound for each driver.

        Starting compound = compound on lap 1 (before any pit stops).
        This is a categorical feature: SOFT, MEDIUM, HARD, INTERMEDIATE, WET.
        """
        try:
            first_laps = laps[laps["LapNumber"] == 1][["Driver", "Compound"]]
            return dict(zip(first_laps["Driver"], first_laps["Compound"].fillna("UNKNOWN")))
        except Exception:
            return {}

    # =========================================================================
    # Historical features (computed after full dataset is collected)
    # =========================================================================

    def _add_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds features that require looking at PAST races.

        These CANNOT be computed during the race loop because we need
        the full dataset to look backwards. Computed here in one pass.

        Features added:
          driver_recent_avg_pos  → driver's avg finish over last 5 races
          constructor_dnf_rate   → team DNF rate over last 10 races

        CRITICAL: For each race, we only look at races BEFORE it.
                  Never include the current race in the window!
                  That would be data leakage.
        """
        df = df.sort_values(["year", "round"]).reset_index(drop=True)

        # Work through races in chronological order
        # For each race, compute the window from all PRECEDING rows
        driver_recent_pos  = []
        constructor_dnf_rt = []

        for i, row in df.iterrows():
            past = df.loc[:i - 1]   # everything BEFORE this race

            # --- Driver recent average position (last 5 races) ---
            driver_past = past[past["driver"] == row["driver"]]
            if len(driver_past) >= 2:
                # Only finished races (not DNF sentinel) for avg position
                finished_past = driver_past[driver_past["finish_position"] < DNF_SENTINEL]
                avg_pos = finished_past["finish_position"].tail(5).mean()
            else:
                avg_pos = np.nan   # not enough history → model will impute

            driver_recent_pos.append(round(avg_pos, 2) if not np.isnan(avg_pos) else np.nan)

            # --- Constructor DNF rate (last 10 races) ---
            team_past = past[past["team"] == row["team"]]
            if len(team_past) >= 3:
                recent_team = team_past.tail(10)
                dnf_rate = recent_team["is_dnf"].mean()
            else:
                dnf_rate = np.nan

            constructor_dnf_rt.append(round(dnf_rate, 4) if not np.isnan(dnf_rate) else np.nan)

        df["driver_recent_avg_pos"] = driver_recent_pos
        df["constructor_dnf_rate"]  = constructor_dnf_rt

        return df
