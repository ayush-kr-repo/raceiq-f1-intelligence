import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DNF_SENTINEL,
    MODELS_DIR,
    OPTUNA_N_TRIALS,
    OUTPUT_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TWO_STAGE_DNF_THRESHOLD,
    XGB_PARAM_SPACE,
)

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class WeightedEnsembleRegressor:
    """Weighted average of multiple trained regressors."""

    def __init__(self, models: dict, weights: dict):
        self.models = models
        self.weights = weights

    def predict(self, X):
        preds = np.zeros(X.shape[0], dtype=float)
        for name, model in self.models.items():
            preds += self.weights[name] * model.predict(X)
        return preds


class TwoStageRacePredictor:
    """Use calibrated probabilities to gate and adjust finish-position regression."""

    def __init__(self, regressor, dnf_model, podium_model, points_model, threshold: float = TWO_STAGE_DNF_THRESHOLD):
        self.regressor = regressor
        self.dnf_model = dnf_model
        self.podium_model = podium_model
        self.points_model = points_model
        self.threshold = threshold

    def predict(self, X):
        base = self.regressor.predict(X)
        details = self.predict_components(X)
        adjusted = base.copy().astype(float)
        adjusted += details["dnf_probability"] * 4.0
        adjusted -= details["podium_probability"] * 2.25
        adjusted -= details["points_probability"] * 0.9
        dnf_mask = details["dnf_probability"] >= self.threshold
        adjusted[dnf_mask] = np.maximum(adjusted[dnf_mask], 16.0 + 5.0 * details["dnf_probability"][dnf_mask])
        return adjusted

    def predict_components(self, X):
        return {
            "base_prediction": self.regressor.predict(X),
            "dnf_probability": self.dnf_model.predict_proba(X)[:, 1],
            "podium_probability": self.podium_model.predict_proba(X)[:, 1],
            "points_probability": self.points_model.predict_proba(X)[:, 1],
        }


class RaceRankingBlend:
    """Ranking-aware blended score: lower is better inside a race."""

    def __init__(self, regressors: dict, weights: dict, dnf_model, podium_model, points_model):
        self.regressors = regressors
        self.weights = weights
        self.dnf_model = dnf_model
        self.podium_model = podium_model
        self.points_model = points_model

    def predict(self, X):
        score = np.zeros(X.shape[0], dtype=float)
        for name, model in self.regressors.items():
            score += self.weights[name] * model.predict(X)
        score += 4.5 * self.dnf_model.predict_proba(X)[:, 1]
        score -= 2.5 * self.podium_model.predict_proba(X)[:, 1]
        score -= 1.25 * self.points_model.predict_proba(X)[:, 1]
        return score


class F1ModelTrainer:
    """Train base regressors plus richer race-prediction wrappers."""

    def train(self, splits: dict) -> dict:
        X_train = splits["X_train"]
        y_train = splits["y_train"]
        X_val = splits["X_val"]
        y_val = splits["y_val"]
        meta_train = splits["meta_train"]

        base_regressors = {
            "lr": self._train_linear(X_train, y_train, X_val, y_val),
            "rf": self._train_rf(X_train, y_train, X_val, y_val),
            "xgb": self._train_xgb(X_train, y_train, X_val, y_val),
        }
        validation_scores = {
            name: self._regression_summary(model, X_val, y_val)
            for name, model in base_regressors.items()
        }
        base_weights = self._ensemble_weights(validation_scores)

        ensemble_model = WeightedEnsembleRegressor(base_regressors, base_weights)
        validation_scores["ensemble"] = self._regression_summary(ensemble_model, X_val, y_val)

        classifiers = self._train_aux_models(X_train, meta_train["is_dnf"].to_numpy(), y_train)

        finisher_regressor = self._train_finisher_regressor(
            X_train,
            y_train,
            meta_train["is_dnf"].to_numpy(),
            X_val,
            y_val,
        )
        validation_scores["finisher_regressor"] = self._regression_summary(finisher_regressor, X_val, y_val)

        two_stage_model = TwoStageRacePredictor(
            regressor=finisher_regressor,
            dnf_model=classifiers["dnf"],
            podium_model=classifiers["podium"],
            points_model=classifiers["points"],
        )
        validation_scores["two_stage"] = self._regression_summary(two_stage_model, X_val, y_val)

        rank_weights = self._ensemble_weights({name: validation_scores[name] for name in ("lr", "rf", "xgb")})
        rankblend_model = RaceRankingBlend(
            regressors=base_regressors,
            weights=rank_weights,
            dnf_model=classifiers["dnf"],
            podium_model=classifiers["podium"],
            points_model=classifiers["points"],
        )
        validation_scores["rankblend"] = self._regression_summary(rankblend_model, X_val, y_val)

        regressors = {
            **base_regressors,
            "ensemble": ensemble_model,
            "finisher_regressor": finisher_regressor,
            "two_stage": two_stage_model,
            "rankblend": rankblend_model,
        }

        for name, model in regressors.items():
            joblib.dump(model, MODELS_DIR / f"{name}_model.pkl")
        for name, model in classifiers.items():
            joblib.dump(model, MODELS_DIR / f"{name}_classifier.pkl")

        summary = {
            "validation_scores": validation_scores,
            "ensemble_weights": base_weights,
            "ranking_weights": rank_weights,
            "feature_count": int(X_train.shape[1]),
            "models": sorted(regressors.keys()),
            "classifiers": sorted(classifiers.keys()),
            "probability_calibration": "sigmoid",
            "best_validation_model": min(validation_scores, key=lambda name: validation_scores[name]["mae"]),
        }
        joblib.dump(summary, OUTPUT_DIR / "training_summary.pkl")
        (REPORTS_DIR / "validation_scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return {
            "regressors": regressors,
            "classifiers": classifiers,
            "validation_scores": validation_scores,
            "ensemble_weights": base_weights,
            "ranking_weights": rank_weights,
        }

    def _train_linear(self, X_train, y_train, X_val, y_val):
        log.info("[1/5] Training Ridge regressor...")
        best_alpha, best_mae = 1.0, float("inf")
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
            model.fit(X_train, y_train)
            preds = self._clip_predictions(model.predict(X_val))
            mae = mean_absolute_error(y_val, preds)
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha
        final_model = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
        final_model.fit(X_train, y_train)
        return final_model

    def _train_rf(self, X_train, y_train, X_val, y_val):
        log.info("[2/5] Training Random Forest regressor...")
        best_params, best_mae = {}, float("inf")
        for n_estimators in [200, 400]:
            for max_depth in [10, 20, None]:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=3,
                    max_features=0.55,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                preds = self._clip_predictions(model.predict(X_val))
                mae = mean_absolute_error(y_val, preds)
                if mae < best_mae:
                    best_mae = mae
                    best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
        final_model = RandomForestRegressor(
            **best_params,
            min_samples_leaf=3,
            max_features=0.55,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        final_model.fit(X_train, y_train)
        return final_model

    def _train_xgb(self, X_train, y_train, X_val, y_val):
        log.info("[3/5] Training XGBoost regressor with Optuna (%s trials)...", OPTUNA_N_TRIALS)
        tscv = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", *XGB_PARAM_SPACE["n_estimators"]),
                "max_depth": trial.suggest_int("max_depth", *XGB_PARAM_SPACE["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *XGB_PARAM_SPACE["learning_rate"], log=True),
                "subsample": trial.suggest_float("subsample", *XGB_PARAM_SPACE["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *XGB_PARAM_SPACE["colsample_bytree"]),
                "min_child_weight": trial.suggest_int("min_child_weight", *XGB_PARAM_SPACE["min_child_weight"]),
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
                "tree_method": "hist",
            }
            fold_maes = []
            for train_idx, val_idx in tscv.split(X_train):
                model = XGBRegressor(**params)
                model.fit(
                    X_train[train_idx],
                    y_train[train_idx],
                    eval_set=[(X_train[val_idx], y_train[val_idx])],
                    verbose=False,
                )
                preds = self._clip_predictions(model.predict(X_train[val_idx]))
                fold_maes.append(mean_absolute_error(y_train[val_idx], preds))
            return float(np.mean(fold_maes))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)

        final_model = XGBRegressor(
            **study.best_params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        )
        final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return final_model

    def _train_finisher_regressor(self, X_train, y_train, y_dnf, X_val, y_val):
        log.info("[5/5] Training finisher-only regressor...")
        finish_mask = np.asarray(y_dnf) == 0
        if finish_mask.sum() < 10:
            log.warning("Too few classified finishers for a dedicated finisher regressor; falling back to Ridge.")
            fallback = Ridge(alpha=1.0, random_state=RANDOM_SEED)
            fallback.fit(X_train, y_train)
            return fallback

        X_finish = X_train[finish_mask]
        y_finish = y_train[finish_mask]
        candidates = [
            Ridge(alpha=1.0, random_state=RANDOM_SEED),
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                max_features=0.55,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ]

        best_model = None
        best_mae = float("inf")
        for candidate in candidates:
            candidate.fit(X_finish, y_finish)
            preds = self._clip_predictions(candidate.predict(X_val))
            mae = mean_absolute_error(y_val, preds)
            if mae < best_mae:
                best_mae = mae
                best_model = candidate
        return best_model

    def _train_aux_models(self, X_train, y_dnf, y_finish):
        log.info("[4/5] Training probability models...")
        y_podium = (y_finish <= 3).astype(int)
        y_points = (y_finish <= 10).astype(int)

        dnf_model = self._fit_calibrated_classifier(
            RandomForestClassifier(
                n_estimators=320,
                max_depth=8,
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            X_train,
            y_dnf,
            label="dnf",
        )
        podium_model = self._fit_calibrated_classifier(
            RandomForestClassifier(
                n_estimators=320,
                max_depth=10,
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            X_train,
            y_podium,
            label="podium",
        )
        points_model = self._fit_calibrated_classifier(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            X_train,
            y_points,
            label="points",
        )
        return {"dnf": dnf_model, "podium": podium_model, "points": points_model}

    def _fit_calibrated_classifier(self, estimator, X, y, label: str):
        y = np.asarray(y)
        class_counts = np.bincount(y.astype(int), minlength=2)
        minority_count = int(class_counts.min())

        if len(np.unique(y)) < 2 or minority_count < 3:
            log.warning(
                "Skipping probability calibration for %s because the class distribution is too small: %s",
                label,
                class_counts.tolist(),
            )
            estimator.fit(X, y)
            return estimator

        cv = min(3, minority_count)
        calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)
        calibrated.fit(X, y)
        return calibrated

    def _regression_summary(self, model, X, y):
        preds = self._clip_predictions(model.predict(X))
        return {
            "mae": round(float(mean_absolute_error(y, preds)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 4),
        }

    def _ensemble_weights(self, validation_scores: dict) -> dict:
        inverse_mae = {
            name: 1.0 / max(scores["mae"], 1e-6)
            for name, scores in validation_scores.items()
        }
        total = sum(inverse_mae.values())
        return {name: value / total for name, value in inverse_mae.items()}

    def _clip_predictions(self, preds: np.ndarray) -> np.ndarray:
        return np.clip(np.round(preds), 1, DNF_SENTINEL).astype(int)

    @staticmethod
    def fix_position_uniqueness(predictions: np.ndarray) -> np.ndarray:
        ranks = np.argsort(np.argsort(predictions)) + 1
        return ranks.astype(int)
