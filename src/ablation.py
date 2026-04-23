import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_GROUPS,
    NUMERIC_FEATURES,
    PLOTS_DIR,
    REPORTS_DIR,
)


class FeatureAblationRunner:
    """Run a compact ablation study over grouped feature stacks."""

    def run(self, splits: dict, final_report: dict) -> pd.DataFrame:
        train_df = splits["X_train_df"].copy()
        val_df = splits["X_val_df"].copy()
        y_train = splits["y_train"]
        y_val = splits["y_val"]
        meta_val = splits["meta_val"].copy()

        stages = [
            ("baseline_core", FEATURE_GROUPS["baseline_core"]),
            ("plus_recent_form", FEATURE_GROUPS["baseline_core"] + FEATURE_GROUPS["recent_form_stack"]),
            ("plus_track_weather", FEATURE_GROUPS["baseline_core"] + FEATURE_GROUPS["recent_form_stack"] + FEATURE_GROUPS["track_weather_stack"]),
            ("plus_telemetry_history", FEATURE_GROUPS["baseline_core"] + FEATURE_GROUPS["recent_form_stack"] + FEATURE_GROUPS["track_weather_stack"] + FEATURE_GROUPS["telemetry_history_stack"]),
        ]

        rows = []
        for stage_name, feature_cols in stages:
            metrics = self._evaluate_stage(stage_name, feature_cols, train_df, val_df, y_train, y_val, meta_val)
            rows.append(metrics)

        for model_name in ("ensemble", "two_stage", "rankblend"):
            if model_name in final_report:
                rows.append(
                    {
                        "stage": model_name,
                        "feature_count": int(len(splits["feature_names"])),
                        "mae": float(final_report[model_name]["mae"]),
                        "rmse": float(final_report[model_name]["rmse"]),
                        "spearman_r": float(final_report[model_name]["spearman_r"]),
                    }
                )

        report_df = pd.DataFrame(rows)
        report_df.to_csv(REPORTS_DIR / "ablation_report.csv", index=False)
        (REPORTS_DIR / "ablation_report.json").write_text(report_df.to_json(orient="records", indent=2), encoding="utf-8")
        self._plot_ablation(report_df)
        return report_df

    def _evaluate_stage(self, stage_name, feature_cols, train_df, val_df, y_train, y_val, meta_val):
        available_cols = [col for col in feature_cols if col in train_df.columns]
        if not available_cols:
            return {"stage": stage_name, "feature_count": 0, "mae": np.nan, "rmse": np.nan, "spearman_r": np.nan}

        non_empty_numeric = {col for col in available_cols if train_df[col].notna().any()}
        pipeline = self._build_pipeline(available_cols, non_empty_numeric)
        X_train = pipeline.fit_transform(train_df[available_cols])
        X_val = pipeline.transform(val_df[available_cols])

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        raw_preds = model.predict(X_val)
        preds = np.clip(np.round(raw_preds), 1, 21).astype(int)

        return {
            "stage": stage_name,
            "feature_count": int(len(available_cols)),
            "mae": round(float(np.mean(np.abs(y_val - preds))), 4),
            "rmse": round(float(np.sqrt(np.mean((y_val - preds) ** 2))), 4),
            "spearman_r": round(self._race_spearman(y_val, raw_preds, meta_val), 4),
        }

    def _build_pipeline(self, available_cols, non_empty_numeric):
        numeric = [
            col
            for col in NUMERIC_FEATURES
            if col in available_cols
            and col in non_empty_numeric
        ]
        categorical = [col for col in CATEGORICAL_FEATURES if col in available_cols]
        binary = [col for col in BINARY_FEATURES if col in available_cols]

        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        binary_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric),
                ("cat", categorical_pipeline, categorical),
                ("bin", binary_pipeline, binary),
            ],
            remainder="drop",
        )

    def _race_spearman(self, y_true, raw_preds, meta_val):
        scores = []
        race_df = pd.DataFrame({"race_id": meta_val["race_id"], "actual": y_true, "predicted": raw_preds})
        for _, group in race_df.groupby("race_id"):
            if len(group) < 3:
                continue
            actual_rank = group["actual"].rank(method="first")
            pred_rank = group["predicted"].rank(method="first")
            scores.append(actual_rank.corr(pred_rank, method="spearman"))
        return float(np.nanmean(scores)) if scores else np.nan

    def _plot_ablation(self, report_df: pd.DataFrame):
        if report_df.empty:
            return
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        metrics = [("mae", "#22c55e"), ("rmse", "#f97316"), ("spearman_r", "#8b5cf6")]
        for ax, (metric, color) in zip(axes, metrics):
            ax.bar(report_df["stage"], report_df[metric], color=color)
            ax.set_title(metric.upper())
            ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "ablation_study.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
