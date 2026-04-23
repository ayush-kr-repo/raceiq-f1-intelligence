import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.append(str(Path(__file__).parent.parent))
from config import CACHE_DIR, ENABLE_TELEMETRY_EXPLAINER, PLOTS_DIR, TELEMETRY_DIR

log = logging.getLogger(__name__)


class F1Explainer:
    """Generate rich post-training visual explainability outputs."""

    def generate(self, report: dict, splits: dict):
        predictions_path = PLOTS_DIR.parent / "reports" / "test_predictions.csv"
        if not predictions_path.exists():
            log.warning("Prediction report not found; skipping explainer outputs.")
            return

        predictions_df = pd.read_csv(predictions_path)
        full_df = splits["full_df"]
        best_model = min(report, key=lambda name: report[name]["mae"])

        self._plot_latest_race_prediction_dashboard(predictions_df, best_model)
        self._plot_driver_form_trends(full_df)
        self._plot_driver_strength_radar(full_df, predictions_df, best_model)

        if ENABLE_TELEMETRY_EXPLAINER:
            self._plot_telemetry_and_strategy(predictions_df, best_model)

    def _plot_latest_race_prediction_dashboard(self, predictions_df: pd.DataFrame, best_model: str):
        latest = predictions_df.sort_values(["year", "round"]).iloc[-1]
        race_id = latest["race_id"]
        race_df = predictions_df[predictions_df["race_id"] == race_id].copy()
        race_df = race_df.sort_values(f"pred_{best_model}").reset_index(drop=True)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        top3 = race_df.sort_values("podium_probability", ascending=False).head(3)
        axes[0].barh(top3["driver"], top3["podium_probability"], color=["#d62828", "#f77f00", "#fcbf49"])
        axes[0].invert_yaxis()
        axes[0].set_xlim(0, 1)
        axes[0].set_title(f"Predicted top 3 support for {race_id}")
        axes[0].set_xlabel("Podium probability")

        compare = race_df.head(10).copy()
        x = np.arange(len(compare))
        axes[1].plot(x, compare["actual_finish"], marker="o", label="Actual", color="#1d3557")
        axes[1].plot(x, compare[f"pred_{best_model}"], marker="o", label=f"Predicted ({best_model})", color="#e63946")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(compare["driver"], rotation=45)
        axes[1].set_ylabel("Finish position")
        axes[1].set_title("Actual vs predicted finish order")
        axes[1].invert_yaxis()
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "latest_race_prediction_dashboard.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_driver_form_trends(self, full_df: pd.DataFrame):
        trend_df = (
            full_df[full_df["finish_position"] < 21]
            .groupby(["year", "driver"])["finish_position"]
            .mean()
            .reset_index()
        )
        top_drivers = (
            trend_df[trend_df["year"] == trend_df["year"].max()]
            .sort_values("finish_position")
            .head(6)["driver"]
            .tolist()
        )
        trend_df = trend_df[trend_df["driver"].isin(top_drivers)]

        fig, ax = plt.subplots(figsize=(11, 6))
        for driver, group in trend_df.groupby("driver"):
            ax.plot(group["year"], group["finish_position"], marker="o", linewidth=2, label=driver)
        ax.set_title("Season trend chart for leading drivers")
        ax.set_xlabel("Season")
        ax.set_ylabel("Average finish position")
        ax.invert_yaxis()
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "season_trend_driver_form.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_driver_strength_radar(self, full_df: pd.DataFrame, predictions_df: pd.DataFrame, best_model: str):
        latest_race_id = predictions_df.sort_values(["year", "round"]).iloc[-1]["race_id"]
        latest_rows = predictions_df[predictions_df["race_id"] == latest_race_id].nsmallest(3, f"pred_{best_model}")
        feature_rows = full_df[full_df["race_id"] == latest_race_id].set_index("driver")

        radar_features = [
            "driver_recent_form_score",
            "track_specific_driver_score",
            "street_circuit_score",
            "quali_to_race_conversion",
            "constructor_recent_form_score",
        ]
        labels = ["Recent form", "Track fit", "Street fit", "Race craft", "Team form"]
        angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
        angles += angles[:1]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        for _, row in latest_rows.iterrows():
            if row["driver"] not in feature_rows.index:
                continue
            values = feature_rows.loc[row["driver"], radar_features].astype(float).fillna(0).to_numpy()
            values = np.clip(values, 0, np.nanmax(values) if np.nanmax(values) > 0 else 1)
            max_value = np.nanmax(values) if np.nanmax(values) > 0 else 1
            scaled = (values / max_value).tolist()
            scaled += scaled[:1]
            ax.plot(angles, scaled, linewidth=2, label=row["driver"])
            ax.fill(angles, scaled, alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        ax.set_title("Predicted podium driver strengths")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "driver_strength_radar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_telemetry_and_strategy(self, predictions_df: pd.DataFrame, best_model: str):
        try:
            import fastf1
        except ModuleNotFoundError:
            log.warning("FastF1 not installed; skipping telemetry explainer.")
            return

        latest_race = predictions_df.sort_values(["year", "round"]).iloc[-1]
        race_id = latest_race["race_id"]
        race_df = predictions_df[predictions_df["race_id"] == race_id].nsmallest(3, f"pred_{best_model}")
        year = int(race_df["year"].iloc[0])
        round_number = int(race_df["round"].iloc[0])
        drivers = race_df["driver"].tolist()

        try:
            fastf1.Cache.enable_cache(str(CACHE_DIR))
            session = fastf1.get_session(year, round_number, "R")
            session.load(telemetry=True, weather=True, messages=False)
        except Exception as exc:
            log.warning(f"Telemetry explainer skipped for {race_id}: {exc}")
            return

        self._plot_tyre_degradation(session, drivers, race_id)
        self._plot_strategy_timeline(session, drivers, race_id)
        self._plot_telemetry_overlay(session, drivers, race_id)

    def _plot_tyre_degradation(self, session, drivers: list[str], race_id: str):
        fig, ax = plt.subplots(figsize=(11, 6))
        for driver in drivers:
            laps = session.laps.pick_driver(driver)
            laps = laps[laps["LapTime"].notna()].copy()
            if laps.empty:
                continue
            laps["lap_seconds"] = laps["LapTime"].dt.total_seconds()
            ax.plot(laps["LapNumber"], laps["lap_seconds"], marker="o", markersize=3, linewidth=1.6, label=driver)
        ax.set_title(f"Tyre degradation comparison for predicted top 3 ({race_id})")
        ax.set_xlabel("Lap number")
        ax.set_ylabel("Lap time (s)")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(TELEMETRY_DIR / f"tyre_degradation_{race_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_strategy_timeline(self, session, drivers: list[str], race_id: str):
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = {
            "SOFT": "#ff595e",
            "MEDIUM": "#ffca3a",
            "HARD": "#adb5bd",
            "INTERMEDIATE": "#00b4d8",
            "WET": "#4361ee",
            "UNKNOWN": "#6c757d",
        }
        for y_pos, driver in enumerate(drivers):
            laps = session.laps.pick_driver(driver)
            if laps.empty or "Stint" not in laps.columns:
                continue
            for _, stint in laps.groupby("Stint"):
                start_lap = int(stint["LapNumber"].min())
                end_lap = int(stint["LapNumber"].max())
                compound = str(stint["Compound"].dropna().iloc[0]) if stint["Compound"].notna().any() else "UNKNOWN"
                ax.barh(
                    y=y_pos,
                    width=end_lap - start_lap + 1,
                    left=start_lap,
                    color=colors.get(compound, "#6c757d"),
                    edgecolor="white",
                    height=0.55,
                )
        ax.set_yticks(range(len(drivers)))
        ax.set_yticklabels(drivers)
        ax.set_xlabel("Lap")
        ax.set_title(f"Stint strategy timeline ({race_id})")
        plt.tight_layout()
        plt.savefig(TELEMETRY_DIR / f"strategy_timeline_{race_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_telemetry_overlay(self, session, drivers: list[str], race_id: str):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        for driver in drivers:
            fastest = session.laps.pick_driver(driver).pick_fastest()
            if fastest is None:
                continue
            try:
                car_data = fastest.get_car_data().add_distance()
            except Exception:
                continue
            axes[0].plot(car_data["Distance"], car_data["Speed"], label=driver)
            axes[1].plot(car_data["Distance"], car_data["RPM"], label=driver)
            axes[2].plot(car_data["Distance"], car_data["Throttle"], label=driver)

        axes[0].set_ylabel("Speed")
        axes[0].set_title(f"Telemetry overlay for predicted top 3 ({race_id})")
        axes[1].set_ylabel("RPM")
        axes[2].set_ylabel("Throttle")
        axes[2].set_xlabel("Distance")
        axes[0].legend(ncol=3)
        for ax in axes:
            ax.grid(alpha=0.25)

        plt.tight_layout()
        plt.savefig(TELEMETRY_DIR / f"telemetry_overlay_{race_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
