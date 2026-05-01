# RaceIQ : F1 Race Intelligence Platform

An end-to-end Formula 1 machine learning platform that predicts race outcomes, explains the predictions, and visualizes race dynamics through an interactive Streamlit dashboard.

## What This Project Does

- Collects or reuses historical FastF1 race data.
- Engineers race, form, track, reliability, and strategy features.
- Trains multiple prediction models plus DNF, podium, and points probability models.
- Combines them into ranking-aware and two-stage race predictors.
- Generates explainability reports and telemetry-backed visuals.
- Serves an interactive Streamlit dashboard for race-level exploration.

## Results

| Model | MAE | RMSE | Top-3 Accuracy | Top-10 Accuracy | Spearman r |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.4447 | 1.5755 | 98.61% | 96.25% | 0.9667 |
| Ensemble | 0.7161 | 1.5848 | 98.61% | 96.25% | 0.9706 |
| Two-stage | 0.8163 | 1.7127 | 98.61% | 97.92% | 0.9748 |
| Random Forest | 1.0271 | 1.8504 | 94.44% | 95.83% | 0.9539 |

- MAE is measured in finishing positions, so the best model was off by less than half a finishing position on average.

Dataset: 2,979 driver-race records across 149 races from 2018-2024.  
Test set: 479 driver-race records across the 2024 season.  
Feature count: 78 engineered features.  
Leakage audit: passed across 2,979 audited rows with 0 detected violations.

## Key Technical Work

- Built temporal train, validation, and test splits to reduce race-history leakage.
- Engineered 78 features covering driver form, constructor form, track fit, reliability, weather sensitivity, DNF risk, and strategy signals.
- Compared Ridge, Random Forest, XGBoost, ensemble, finisher-regressor, two-stage, and rank-blend approaches.
- Added SHAP-based global and per-prediction explainability.
- Ran ablation analysis showing MAE improvement from 4.2227 baseline to 0.7161 with the full ensemble pipeline.

## Architecture

### Predictor Layer

- `src/collector.py`
  Collects raw race-level features from FastF1.
- `src/feature_builder.py`
  Adds historical and contextual features such as recent form, track fit, DNF risk, and strategy strength.
- `src/preprocessor.py`
  Enriches raw data, applies time-based train/validation/test split, and builds preprocessing pipelines.
- `src/trainer.py`
  Trains Ridge, Random Forest, XGBoost, ensemble, rank-blend, and two-stage predictors.
- `src/evaluator.py`
  Measures performance, exports detailed prediction tables, and writes explainability artifacts.

### Explainer Layer

- `src/explainer.py`
  Produces race dashboards, telemetry overlays, tyre degradation plots, strategy timelines, and form visuals.
- `app.py`
  Interactive Streamlit product dashboard with model comparison, race pulse, telemetry bay, and explanations.

## Main ML Features

- Time-aware split to avoid leakage.
- Advanced historical features:
  - Driver recent form
  - Constructor recent form
  - Track-specific driver score
  - Wet performance score
  - Street circuit score
  - DNF risk score
  - Strategy score
  - Speed-trap and degradation history
- Probability models:
  - DNF probability
  - Podium probability
  - Points probability
- Prediction strategies:
  - Baseline regressors
  - Weighted ensemble
  - Two-stage race predictor
  - Ranking-aware blend
- Explainability:
  - SHAP global summary
  - Per-prediction top feature contributions

## Dashboard Highlights

- Race selector and model selector.
- Model comparison charts.
- Predicted podium and probability views.
- Telemetry, tyre degradation, and strategy visualizations.
- SHAP and per-driver explanation records.

## How To Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify the environment

```bash
python scripts/verify_setup.py
```

This writes a machine-readable report to `outputs/verify/setup_report.json`.

### 3. Reuse the included dataset

```bash
python run.py --skip-collect
```

This public repo keeps the source, tests, dataset, and sample visual outputs.
Large generated model binaries and machine-specific run caches are intentionally omitted to keep the repository cleaner.

### 4. Recollect from FastF1

```bash
python run.py --collect-only --seasons 2022
python run.py --collect-only --seasons 2023
python run.py --collect-only --seasons 2024
python run.py --skip-collect
```

### 5. Launch dashboard

```bash
streamlit run app.py
```

or on Windows:

```bash
launch_dashboard.bat
```

## Output Structure

- `outputs/models/`
  generated locally after training; omitted from the public repo by default
- `outputs/metrics_report.json`
- `outputs/reports/metrics_report.json`
- `outputs/shap_summary.png`
- `outputs/shap_waterfall_example.png`
- `outputs/plots/`
- `outputs/reports/test_predictions.csv`
- `outputs/reports/latest_race_summary.json`
- `outputs/reports/prediction_explanations.csv`
- `outputs/reports/leakage_audit.json`
- `outputs/reports/ablation_report.csv`
- `outputs/verify/setup_report.json`
- `outputs/telemetry/`
- `outputs/runs/`

## Public Repo Notes

- sample plots, telemetry visuals, and reports are included so the dashboard still has meaningful demo content
- heavy `.pkl` model artifacts are excluded from the public version and can be regenerated with `python run.py --skip-collect`
- machine-specific files like `run.log`, setup reports, and old run manifests are not part of the public version

## Testing

Run the verification script:

```bash
python scripts/verify_setup.py
```

Run the full test suite:

```bash
python -m pytest
```

or the smoke tests only:

```bash
python -m unittest tests.test_project_smoke
```

## Why This Project Matters

- Combines machine learning, data engineering, explainability, domain understanding, and UI/product work
- Uses real race data instead of toy datasets
- Treats race prediction as both regression and ranking
- Separates predictor logic from explainer logic

## Current Limitations

- FastF1 session quality varies by weekend and can require skips
- some telemetry comparisons are post-race explainability outputs, not pre-race features
- ranking-aware logic is heuristic rather than a full learning-to-rank implementation
- probabilities are calibrated with lightweight sigmoid calibration, but rare-event reliability can still improve

## Future Upgrades

- public deployment
- true learn-to-rank model
- richer telemetry-derived historical features
- automated CI and packaging
