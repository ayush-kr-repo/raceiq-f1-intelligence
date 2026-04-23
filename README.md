# RaceIQ : F1 Race Intelligence Platform

An end-to-end Formula 1 machine learning platform that predicts race outcomes, explains the predictions, and visualizes race dynamics through a premium interactive dashboard.

## What This Project Does

- collects or reuses historical FastF1 race data
- engineers race, form, track, reliability, and strategy features
- trains multiple prediction models plus DNF/podium/points probability models
- combines them into ranking-aware and two-stage race predictors
- generates explainability reports and telemetry-backed visuals
- serves a neon-styled Streamlit dashboard for exploration

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

- time-aware split to avoid leakage
- advanced historical features:
  - driver recent form
  - constructor recent form
  - track-specific driver score
  - wet performance score
  - street circuit score
  - DNF risk score
  - strategy score
  - speed-trap and degradation history
- probability models:
  - DNF probability
  - podium probability
  - points probability
- prediction strategies:
  - baseline regressors
  - weighted ensemble
  - two-stage race predictor
  - ranking-aware blend
- explainability:
  - SHAP global summary
  - per-prediction top feature contributions

## Dashboard Highlights

- premium neon visual design
- animated hero and glassmorphism cards
- race selector and model selector
- model lab with interactive charts
- predicted podium and probability pressure maps
- telemetry bay with tyre degradation and strategy outputs
- explanation tab with SHAP and per-driver explanation records

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

## Why This Project Is Strong

- combines machine learning, data engineering, explainability, domain understanding, and UI/product work
- uses real race data instead of toy datasets
- treats race prediction as both regression and ranking
- separates predictor logic from explainer logic

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
