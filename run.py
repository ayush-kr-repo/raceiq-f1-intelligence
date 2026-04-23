import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))

from config import ALL_SEASONS, DATA_DIR, OUTPUT_DIR, REPORTS_DIR
from src.ablation import FeatureAblationRunner
from src.evaluator import F1Evaluator
from src.experiment_tracker import ExperimentTracker
from src.explainer import F1Explainer
from src.preprocessor import F1Preprocessor
from src.trainer import F1ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "run.log"),
    ],
)
log = logging.getLogger(__name__)

RAW_DATA_PATH = DATA_DIR / "raw_data.csv"


def main():
    parser = argparse.ArgumentParser(description="F1 prediction and explainability pipeline")
    parser.add_argument("--skip-collect", action="store_true", help="Use the existing raw_data.csv")
    parser.add_argument("--collect-only", action="store_true", help="Collect data, update raw_data.csv, and stop")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=ALL_SEASONS,
        help="Seasons to collect from FastF1",
    )
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("F1 predictor v2: predictor + explainer pipeline")
    log.info("=" * 70)

    from config import RUNS_DIR, TEST_YEAR, TRAIN_YEARS, VAL_YEAR

    tracker = ExperimentTracker(RUNS_DIR)
    tracker.log_params(
        {
            "train_years": TRAIN_YEARS,
            "val_year": VAL_YEAR,
            "test_year": TEST_YEAR,
            "requested_seasons": args.seasons,
            "skip_collect": args.skip_collect,
        }
    )

    if not args.skip_collect:
        _collect_data(args.seasons)
        if args.collect_only:
            tracker.complete(status="collect_only")
            log.info("Collection finished. Exiting because --collect-only was set.")
            return
    else:
        _ensure_raw_data_exists()

    preprocessor = F1Preprocessor()
    splits = preprocessor.fit_transform(str(RAW_DATA_PATH))
    tracker.log_params(
        {
            "feature_count": len(splits["feature_names"]),
            "train_rows": int(splits["X_train"].shape[0]),
            "val_rows": int(splits["X_val"].shape[0]),
            "test_rows": int(splits["X_test"].shape[0]),
            "feature_schema_version": abs(hash(tuple(splits["feature_names"]))),
            "leakage_audit_status": splits["leakage_audit"]["status"],
        }
    )

    trainer = F1ModelTrainer()
    model_registry = trainer.train(splits)
    tracker.log_params(
        {
            "trained_models": sorted(model_registry["regressors"].keys()),
            "trained_classifiers": sorted(model_registry["classifiers"].keys()),
        }
    )

    evaluator = F1Evaluator()
    report = evaluator.evaluate(model_registry, splits)
    ablation_report = FeatureAblationRunner().run(splits, report)

    explainer = F1Explainer()
    explainer.generate(report, splits)

    summary = {
        "available_outputs": sorted([path.name for path in OUTPUT_DIR.glob("*")]),
        "report_models": list(report.keys()),
        "run_id": tracker.run_id,
        "leakage_audit_status": splits["leakage_audit"]["status"],
        "ablation_rows": int(len(ablation_report)),
    }
    (REPORTS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tracker.log_metrics({f"{model_name}_mae": metrics["mae"] for model_name, metrics in report.items()})
    tracker.log_metrics({f"{model_name}_spearman": metrics["spearman_r"] for model_name, metrics in report.items()})
    tracker.log_metrics({"ablation_rows": int(len(ablation_report))})
    tracker.log_artifact(str(REPORTS_DIR / "run_summary.json"))
    tracker.log_artifact(str(REPORTS_DIR / "test_predictions.csv"))
    tracker.log_artifact(str(OUTPUT_DIR / "metrics_report.json"))
    tracker.log_artifact(str(REPORTS_DIR / "leakage_audit.json"))
    tracker.log_artifact(str(REPORTS_DIR / "ablation_report.csv"))
    tracker.complete()

    best_model = min(report, key=lambda model_name: report[model_name]["mae"])
    log.info("Best test MAE model: %s", best_model.upper())
    for model_name, metrics in report.items():
        log.info(
            "%s | MAE=%.4f RMSE=%.4f Top3=%.1f%% Top10=%.1f%% Spearman=%.4f",
            model_name.upper(),
            metrics["mae"],
            metrics["rmse"],
            metrics["top3_accuracy"] * 100,
            metrics["top10_accuracy"] * 100,
            metrics["spearman_r"],
        )

    log.info("Pipeline complete. Outputs written to %s", OUTPUT_DIR)


def _collect_data(seasons: list[int]):
    log.info("[STEP 1] Collecting FastF1 data for seasons: %s", seasons)
    try:
        from src.collector import F1DataCollector
    except ModuleNotFoundError as exc:
        log.error("FastF1 dependencies are missing. Install requirements.txt before collecting data.")
        raise SystemExit(1) from exc

    collector = F1DataCollector()
    df = collector.collect(seasons=seasons)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists():
        existing_df = pd.read_csv(RAW_DATA_PATH)
        df = pd.concat([existing_df, df], ignore_index=True)
        df = df.drop_duplicates(subset=["race_id", "driver"], keep="last")
        df = df.sort_values(["year", "round", "driver"], kind="stable").reset_index(drop=True)

    df.to_csv(RAW_DATA_PATH, index=False)
    log.info("Raw data saved to %s with shape %s", RAW_DATA_PATH, df.shape)


def _ensure_raw_data_exists():
    if not RAW_DATA_PATH.exists():
        log.error("raw_data.csv not found at %s", RAW_DATA_PATH)
        log.error("Run without --skip-collect first, or copy the dataset into the data folder.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
