import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    REPORTS_DIR,
    TARGET_COL,
    TEST_YEAR,
    TRAIN_YEARS,
    VAL_YEAR,
)
from src.feature_builder import F1FeatureBuilder
from src.leakage_audit import LeakageAuditor

log = logging.getLogger(__name__)


class F1Preprocessor:
    """Load, enrich, split, and preprocess the tabular dataset."""

    def __init__(self):
        self.pipeline = None
        self.feature_names = []
        self._available_cols = set()
        self._non_empty_numeric_cols = set()
        self.feature_builder = F1FeatureBuilder()
        self.leakage_auditor = LeakageAuditor()

    def fit_transform(self, data_path: str) -> dict:
        df = pd.read_csv(data_path)
        log.info(f"Loaded {len(df)} rows from {data_path}")

        df = self.feature_builder.build(df)
        enriched_path = REPORTS_DIR / "enriched_raw_data.csv"
        df.to_csv(enriched_path, index=False)
        log.info(f"Enriched feature dataset saved to {enriched_path}")
        leakage_audit = self.leakage_auditor.audit(df)
        log.info("Leakage audit status: %s", leakage_audit["status"])

        train_df, val_df, test_df = self._time_split(df)
        X_train, y_train = self._split_xy(train_df)
        X_val, y_val = self._split_xy(val_df)
        X_test, y_test = self._split_xy(test_df)

        self._available_cols = set(X_train.columns)
        self._non_empty_numeric_cols = {col for col in X_train.columns if X_train[col].notna().any()}
        self.pipeline = self._build_pipeline()

        X_train_proc = self.pipeline.fit_transform(X_train)
        X_val_proc = self.pipeline.transform(X_val)
        X_test_proc = self.pipeline.transform(X_test)
        self.feature_names = self._get_feature_names()

        pipeline_path = OUTPUT_DIR / "preprocessor.pkl"
        joblib.dump(self.pipeline, pipeline_path)
        log.info(f"Pipeline saved to {pipeline_path}")

        meta_cols = [
            "race_id",
            "driver",
            "team",
            "year",
            "round",
            "gp_name",
            "circuit",
            "is_dnf",
            "season_phase",
        ]

        return {
            "X_train": X_train_proc,
            "y_train": y_train.to_numpy(),
            "X_val": X_val_proc,
            "y_val": y_val.to_numpy(),
            "X_test": X_test_proc,
            "y_test": y_test.to_numpy(),
            "feature_names": self.feature_names,
            "X_train_df": X_train.reset_index(drop=True),
            "X_val_df": X_val.reset_index(drop=True),
            "X_test_df": X_test.reset_index(drop=True),
            "meta_train": train_df[meta_cols].reset_index(drop=True),
            "meta_val": val_df[meta_cols].reset_index(drop=True),
            "meta_test": test_df[meta_cols].reset_index(drop=True),
            "full_df": df.reset_index(drop=True),
            "leakage_audit": leakage_audit,
        }

    def _time_split(self, df: pd.DataFrame):
        if TARGET_COL not in df.columns:
            raise ValueError(f"Required target column '{TARGET_COL}' not found.")
        ordered = df.sort_values(["year", "round", "race_id", "driver"], kind="stable").reset_index(drop=True)

        train = ordered[ordered["year"].isin(TRAIN_YEARS)].copy()
        val = ordered[ordered["year"] == VAL_YEAR].copy()
        test = ordered[ordered["year"] == TEST_YEAR].copy()

        if train.empty or val.empty or test.empty:
            raise ValueError("Train, validation, and test splits must all contain rows.")

        log.info(f"Train: {len(train)} rows | Val: {len(val)} | Test: {len(test)}")
        return train, val, test

    def _split_xy(self, df: pd.DataFrame):
        feature_cols = [col for col in (NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES) if col in df.columns]
        if not feature_cols:
            raise ValueError("No configured feature columns were found.")
        return df[feature_cols].copy(), df[TARGET_COL].copy()

    def _build_pipeline(self) -> ColumnTransformer:
        available_numeric = [col for col in NUMERIC_FEATURES if col in self._available_cols and col in self._non_empty_numeric_cols]
        available_cat = [col for col in CATEGORICAL_FEATURES if col in self._available_cols]
        available_binary = [col for col in BINARY_FEATURES if col in self._available_cols]

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
        binary_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="most_frequent"))]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, available_numeric),
                ("cat", categorical_pipeline, available_cat),
                ("bin", binary_pipeline, available_binary),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )

    def _get_feature_names(self) -> list[str]:
        try:
            return list(self.pipeline.get_feature_names_out())
        except Exception:
            return []
