import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use("Agg")

sys.path.append(str(Path(__file__).parent.parent))
from config import DNF_SENTINEL, OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR
from src.trainer import F1ModelTrainer

log = logging.getLogger(__name__)


class F1Evaluator:
    """Evaluate regressors, save prediction tables, and emit dashboard artifacts."""

    def evaluate(self, model_registry: dict, splits: dict) -> dict:
        regressors = model_registry["regressors"]
        classifiers = model_registry["classifiers"]
        X_test = splits["X_test"]
        y_test = splits["y_test"]
        meta = splits["meta_test"].reset_index(drop=True)
        feature_names = splits.get("feature_names", [])

        predictions_df = meta.copy()
        predictions_df["actual_finish"] = y_test
        predictions_df["dnf_probability"] = classifiers["dnf"].predict_proba(X_test)[:, 1]
        predictions_df["podium_probability"] = classifiers["podium"].predict_proba(X_test)[:, 1]
        predictions_df["points_probability"] = classifiers["points"].predict_proba(X_test)[:, 1]
        predictions_df["confidence_score"] = (
            predictions_df["points_probability"] + predictions_df["podium_probability"] - predictions_df["dnf_probability"]
        ).clip(lower=0.0, upper=1.0)

        report = {}
        for name, model in regressors.items():
            raw_preds = model.predict(X_test)
            preds = np.clip(np.round(raw_preds), 1, DNF_SENTINEL).astype(int)
            predictions_df[f"pred_{name}"] = preds
            predictions_df[f"score_{name}"] = raw_preds

            metrics = {
                "mae": self._mae(y_test, preds),
                "rmse": self._rmse(y_test, preds),
            }
            metrics.update(self._compute_race_level_metrics(y_test, raw_preds, meta))
            metrics["residuals"] = self._residual_summary(y_test, preds, model_name=name)
            report[name] = metrics

        shap_explanations = {}
        if "xgb" in regressors:
            shap_explanations = self._generate_shap(regressors["xgb"], X_test, feature_names)

        self._save_prediction_artifacts(predictions_df, shap_explanations)
        self._plot_model_comparison(report)
        self._plot_probability_overview(predictions_df)
        self._plot_finish_scatter(predictions_df)
        self._plot_ranking_comparison(predictions_df)

        report_path = OUTPUT_DIR / "metrics_report.json"
        reports_copy_path = REPORTS_DIR / "metrics_report.json"
        serializable_report = self._make_serializable(report)
        for path in (report_path, reports_copy_path):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(serializable_report, handle, indent=2)
        log.info("Metrics saved to %s and %s", report_path, reports_copy_path)
        return report

    def _save_prediction_artifacts(self, predictions_df: pd.DataFrame, shap_explanations: dict):
        predictions_path = REPORTS_DIR / "test_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        log.info(f"Detailed test predictions saved to {predictions_path}")

        latest_race_id = predictions_df.sort_values(["year", "round"]).iloc[-1]["race_id"]
        latest_race_df = predictions_df[predictions_df["race_id"] == latest_race_id].copy()
        latest_race_df = latest_race_df.sort_values("pred_two_stage")
        latest_summary = {
            "race_id": latest_race_id,
            "strongest_confidence_driver": latest_race_df.sort_values("confidence_score", ascending=False).head(1)[
                ["driver", "confidence_score", "podium_probability", "points_probability", "dnf_probability"]
            ].to_dict(orient="records"),
            "best_predicted_top3": latest_race_df.head(3)[["driver", "pred_two_stage", "podium_probability"]].to_dict(orient="records"),
            "full_table": latest_race_df.to_dict(orient="records"),
        }
        (REPORTS_DIR / "latest_race_summary.json").write_text(json.dumps(self._make_serializable(latest_summary), indent=2), encoding="utf-8")

        if shap_explanations:
            explanation_rows = []
            for idx, explanation in shap_explanations.items():
                row = predictions_df.iloc[idx]
                explanation_rows.append(
                    {
                        "race_id": row["race_id"],
                        "driver": row["driver"],
                        "model": "xgb",
                        "actual_finish": int(row["actual_finish"]),
                        "predicted_finish": int(row["pred_xgb"]),
                        "top_features": explanation["top_features"],
                        "top_impacts": explanation["top_impacts"],
                    }
                )
            pd.DataFrame(explanation_rows).to_csv(REPORTS_DIR / "prediction_explanations.csv", index=False)

    def _mae(self, y_true, y_pred) -> float:
        return round(float(mean_absolute_error(y_true, y_pred)), 4)

    def _rmse(self, y_true, y_pred) -> float:
        return round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)

    def _compute_race_level_metrics(self, y_test, raw_preds, meta) -> dict:
        results_df = pd.DataFrame({"race_id": meta["race_id"].values, "actual": y_test, "predicted_raw": raw_preds})
        top3_scores = []
        top10_scores = []
        spearman_rs = []
        for _, group in results_df.groupby("race_id"):
            actual = group["actual"].to_numpy()
            predicted_unique = F1ModelTrainer.fix_position_uniqueness(group["predicted_raw"].to_numpy().astype(float))
            top3_scores.append(self._topk_set_overlap(actual, predicted_unique, 3))
            top10_scores.append(self._topk_set_overlap(actual, predicted_unique, 10))
            r, _ = self._spearman(actual, predicted_unique)
            if not np.isnan(r):
                spearman_rs.append(r)
        mean_spearman = float(np.mean(spearman_rs)) if spearman_rs else np.nan
        return {
            "top3_accuracy": round(float(np.mean(top3_scores)), 4) if top3_scores else np.nan,
            "top10_accuracy": round(float(np.mean(top10_scores)), 4) if top10_scores else np.nan,
            "spearman_r": round(mean_spearman, 4) if not np.isnan(mean_spearman) else np.nan,
        }

    def _topk_set_overlap(self, actual: np.ndarray, predicted: np.ndarray, k: int) -> float:
        return len(set(np.argsort(predicted)[:k]) & set(np.argsort(actual)[:k])) / k

    def _spearman(self, actual: np.ndarray, predicted: np.ndarray) -> tuple:
        if len(actual) < 3:
            return np.nan, np.nan
        try:
            result = spearmanr(actual, predicted)
            return float(result.statistic), float(result.pvalue)
        except Exception:
            return np.nan, np.nan

    def _residual_summary(self, y_true, y_pred, model_name: str) -> dict:
        residuals = y_true.astype(float) - y_pred.astype(float)
        summary = {
            "mean_residual": round(float(residuals.mean()), 4),
            "std_residual": round(float(residuals.std()), 4),
        }
        self._plot_residuals(residuals, y_true, model_name)
        return summary

    def _plot_residuals(self, residuals, y_true, model_name: str):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(y_true, residuals, alpha=0.45, s=24, color="#2a7de1")
        axes[0].axhline(0, color="#ff5a5f", linestyle="--", linewidth=1.2)
        axes[0].set_xlabel("Actual finish")
        axes[0].set_ylabel("Residual")
        axes[0].set_title(f"{model_name.upper()} residuals vs actual")
        axes[1].hist(residuals, bins=30, color="#46b29d", alpha=0.85, edgecolor="white")
        axes[1].axvline(0, color="#ff5a5f", linestyle="--", linewidth=1.2)
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Residual distribution")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"residual_plot_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_model_comparison(self, report: dict):
        df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "model"})
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        specs = [("mae", "MAE", "#22c55e"), ("rmse", "RMSE", "#f97316"), ("spearman_r", "Spearman", "#8b5cf6")]
        for ax, (metric, title, color) in zip(axes, specs):
            ax.bar(df["model"], df[metric], color=color)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_probability_overview(self, predictions_df: pd.DataFrame):
        latest_race_id = predictions_df.sort_values(["year", "round"]).iloc[-1]["race_id"]
        race_df = predictions_df[predictions_df["race_id"] == latest_race_id].sort_values("podium_probability", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(race_df["driver"], race_df["podium_probability"], color="#8a5cf6")
        ax.invert_yaxis()
        ax.set_xlabel("Predicted podium probability")
        ax.set_title(f"Podium probabilities for {latest_race_id}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "podium_probabilities_latest_race.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_finish_scatter(self, predictions_df: pd.DataFrame):
        if "pred_two_stage" not in predictions_df.columns:
            return
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(predictions_df["actual_finish"], predictions_df["pred_two_stage"], alpha=0.5, color="#16a34a")
        ax.plot([1, DNF_SENTINEL], [1, DNF_SENTINEL], linestyle="--", color="#ff5a5f")
        ax.set_xlabel("Actual finish")
        ax.set_ylabel("Predicted finish (two-stage)")
        ax.set_title("Actual vs predicted finish")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "actual_vs_predicted_two_stage.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_ranking_comparison(self, predictions_df: pd.DataFrame):
        if "pred_rankblend" not in predictions_df.columns:
            return
        latest_race_id = predictions_df.sort_values(["year", "round"]).iloc[-1]["race_id"]
        race_df = predictions_df[predictions_df["race_id"] == latest_race_id].copy()
        race_df = race_df.sort_values("pred_rankblend")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(race_df["driver"], race_df["actual_finish"], marker="o", label="Actual", color="#1d3557")
        ax.plot(race_df["driver"], race_df["pred_rankblend"], marker="o", label="Rank blend", color="#f43f5e")
        ax.invert_yaxis()
        ax.set_title(f"Ranking comparison for {latest_race_id}")
        ax.set_ylabel("Finish position")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "latest_race_rankblend_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _generate_shap(self, xgb_model, X_test, feature_names: list):
        try:
            import shap
        except ModuleNotFoundError:
            log.warning("SHAP not installed; skipping SHAP output.")
            return {}

        explanations = {}
        try:
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer(X_test)

            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values.values,
                X_test,
                feature_names=feature_names if feature_names else None,
                show=False,
                max_display=15,
            )
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap_values[0], max_display=12, show=False)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
            plt.close()

            shap_frame = pd.DataFrame(
                shap_values.values,
                columns=feature_names if feature_names else [f"f{i}" for i in range(shap_values.values.shape[1])],
            )
            shap_frame.to_csv(OUTPUT_DIR / "shap_values_test.csv", index=False)

            for idx, row in shap_frame.iterrows():
                abs_sorted = row.abs().sort_values(ascending=False).head(5)
                explanations[idx] = {
                    "top_features": " | ".join(abs_sorted.index.tolist()),
                    "top_impacts": " | ".join([f"{row[name]:.3f}" for name in abs_sorted.index]),
                }
        except Exception as exc:
            log.warning(f"SHAP generation failed: {exc}")
        return explanations

    def _make_serializable(self, obj):
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
