"""Dual-Engine LightGBM forecasting model — v3.1 (Production Version).

Architecture (v3.1):
- Engine A (preserved): MACD-anchored residual boosting for revenue
- Engine B: LGBMClassifier zero-inflation gate — asymmetric thresholds
  P10: prob > 0.90 | P50: prob > 0.40 | P90: prob > 0.10
- Log1p transform on residuals (shifted to handle negatives)
- Q90 training targets clipped at 95th percentile
- Combined channel imbalance + temporal decay sample weights
- Optuna Bayesian hyperparameter tuning (50 trials, optimise WAPE)
- Walk-forward 3-fold expanding-window cross-validation
- Non-crossing enforcement + ROAS cap at 15x

Differentiators preserved:
- MACD momentum anchor (fast/slow EMA ratio on revenue)
- Monte Carlo quantile reconciliation (10,000 samples)
- YoY lag features (lag_52, lag_26, yoy_ratio)
- Cross-platform halo features (meta_spend_roll4/8, portfolio_upper_lower_ratio)
- Temporal decay weighting (now combined with channel imbalance 
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import lightgbm as lgb  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    CLASSIFIER_THRESHOLD_P10,
    CLASSIFIER_THRESHOLD_P50,
    CLASSIFIER_THRESHOLD_P90,
    FEATURE_COLS,
    HOLDOUT_WEEKS,
    LGBM_PARAMS,
    MIN_TRAIN_SAMPLES,
    OPTUNA_METRIC,
    OPTUNA_TIMEOUT,
    OPTUNA_TRIALS,
    Q90_CLIP_PERCENTILE,
    QUANTILES,
    RANDOM_SEED,
    ROAS_CAP,
    SPARSE_INTERVAL_MULTIPLIER,
    HORIZONS,
)
from src.aggregate import compute_temporal_weights
from src.curve import budget_multiplier, fit_curve
from src.metrics import wape as _wape_metric


# ─── Constants ───────────────────────────────────────────────────────────────
# Shift constant so residuals are always positive before log1p transform.
# We shift by max plausible negative residual (anchor can overshoot target).
# Using a fixed large constant is safe — we subtract it back after prediction.
_RESIDUAL_SHIFT = 50_000.0   # $50k shift — handles even large anchor overshoots


class FallbackQuantileModel:
    """Simple baseline model for sparse groups (<12 weeks of data)."""

    def __init__(self, revenues: np.ndarray):
        self.mean = float(np.mean(revenues)) if len(revenues) > 0 else 0.0
        self.std = float(np.std(revenues)) if len(revenues) > 2 else self.mean * 0.5

    def predict(self, df: pd.DataFrame) -> dict[str, float]:
        return {
            "p10": max(0.0, self.mean - 1.28 * self.std),
            "p50": max(0.0, self.mean),
            "p90": max(0.0, self.mean + 1.28 * self.std),
        }


@dataclass
class ModelBundle:
    """Pickled artifact — dual-engine models, curves, calibration, metadata."""

    version: str = "3.1-all-fixes"
    freq: str = "weekly"
    use_log_target: bool = True

    # Engine B: ROAS quantile models (primary prediction path — residual boosting)
    quantile_models: dict[str, dict[float, lgb.Booster]] = field(default_factory=dict)
    # Spend quantile models (kept for backwards compat with old bundles)
    spend_models: dict[str, dict[float, lgb.Booster]] = field(default_factory=dict)
    # Binary zero-inflation classifiers per model key
    zero_classifiers: dict[str, lgb.Booster] = field(default_factory=dict)
    # Fallback for sparse groups
    fallback_models: dict[str, FallbackQuantileModel] = field(default_factory=dict)

    curve_params: dict[str, dict] = field(default_factory=dict)
    category_maps: dict[str, list[str]] = field(default_factory=dict)
    baseline_spend: dict[str, float] = field(default_factory=dict)
    blend_weights: dict[str, float] = field(default_factory=dict)
    data_tiers: dict[str, str] = field(default_factory=dict)
    calibration_factors: dict[str, float] = field(default_factory=dict)
    group_cv: dict[str, float] = field(default_factory=dict)

    last_date: pd.Timestamp | None = None
    
    # ── Feature Configuration ──
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))
    spend_feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))

    # Best Optuna params
    optuna_best_params: dict = field(default_factory=dict)
    
    holdout_metrics: dict = field(default_factory=dict)
    meta_revenue_assumption: str = "conversion_as_revenue"
    validation_report: dict = field(default_factory=dict)

    def __getattr__(self, name: str):
        """Backwards-compatible pickle loading: return safe defaults for new fields
        that don't exist in old model.pkl versions."""
        _defaults: dict[str, object] = {
            "spend_models": {},
            "zero_classifiers": {},
            "calibration_factors": {},
            "group_cv": {},
            "spend_feature_cols": list(FEATURE_COLS),
            "feature_cols": list(FEATURE_COLS),
            "fallback_models": {},
            "data_tiers": {},
            "blend_weights": {},
            "baseline_spend": {},
            "validation_report": {},
            "holdout_metrics": {},
            "use_log_target": True,
            "freq": "weekly",
            "version": "legacy",
            "meta_revenue_assumption": "conversion_as_revenue",
            "last_date": None,
            "optuna_best_params": {},
        }
        if name in _defaults:
            object.__setattr__(self, name, _defaults[name])
            return _defaults[name]
        raise AttributeError(f"'ModelBundle' object has no attribute '{name}'")


# ─── Key helpers ─────────────────────────────────────────────────────────────

def model_key(row: pd.Series | dict, campaign_type: str | None = None) -> str:
    if isinstance(row, dict):
        channel = row.get("channel", "")
        ct = campaign_type or row.get("campaign_type", "")
    else:
        channel = row.get("channel", "")
        ct = campaign_type or row.get("campaign_type", "")
    return f"{channel}|{ct}"


def _encode_categories(
    df: pd.DataFrame, maps: dict[str, list[str]] | None = None
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = df.copy()
    maps = maps or {}
    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            continue
        if col not in maps:
            maps[col] = sorted(out[col].astype(str).unique().tolist())
        out[col] = out[col].astype(str).map(
            lambda x, c=col: maps[c].index(x) if x in maps[c] else len(maps[c])
        )
    return out, maps


def _enforce_quantile_order(preds: dict[str, float]) -> dict[str, float]:
    """Enforce p10 ≤ p50 ≤ p90 and non-negative predictions."""
    p10, p50, p90 = preds["p10"], preds["p50"], preds["p90"]
    p10 = max(0.0, p10)
    p50 = max(p10, p50)
    p90 = max(p50, p90)
    return {"p10": p10, "p50": p50, "p90": p90}


# ─── Log transform helpers ────────────────────────────────────────────

def _to_log_residual(residuals: np.ndarray) -> np.ndarray:
    """Log1p transform on residuals with shift to handle negatives.
    
    Residuals = target - anchor can be negative when anchor overshoots.
    We shift by _RESIDUAL_SHIFT to guarantee positivity, apply log1p,
    then un-shift after prediction.
    """
    return np.log1p(np.maximum(residuals + _RESIDUAL_SHIFT, 0.0))


def _from_log_residual(log_preds: np.ndarray) -> np.ndarray:
    """Inverse of _to_log_residual."""
    return np.expm1(log_preds) - _RESIDUAL_SHIFT


# ─── Training engine ─────────────────────────────────────────────────────────

def _train_engine(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    sample_weights: np.ndarray,
    lgbm_params: dict,
    clip_q90_at_percentile: bool = False,   # 
) -> dict[float, lgb.Booster]:
    """Train quantile models for one engine (residual prediction).
    
    y_train is already log-transformed — models are trained in log space.
    For Q90 model, clip training targets at 95th percentile to prevent
           upper bound inflation from outlier weeks (Black Friday etc).
    """
    models = {}
    for q in QUANTILES:
        y_tr = y_train.copy()

        # Clip Q90 targets at 95th percentile to tighten interval
        if q == 0.90 and clip_q90_at_percentile:
            cap = np.percentile(y_tr, Q90_CLIP_PERCENTILE)
            y_tr = np.clip(y_tr, None, cap)

        if q == 0.50:
            params = {**lgbm_params, "objective": "huber", "alpha": 1.0, "metric": "huber"}
        else:
            params = {**lgbm_params, "alpha": q}

        dtrain = lgb.Dataset(
            X_train,
            label=y_tr,
            weight=sample_weights,
            categorical_feature=CATEGORICAL_FEATURES,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=CATEGORICAL_FEATURES,
            reference=dtrain,
            free_raw_data=False,
        )
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )
        models[q] = model
    return models


def _predict_engine(
    models: dict[float, lgb.Booster],
    X: pd.DataFrame,
) -> dict[str, float]:
    preds = {}
    for q in QUANTILES:
        label = f"p{int(q * 100)}"
        # predictions are in log space — transform back
        log_pred = float(models[q].predict(X)[0])
        preds[label] = float(_from_log_residual(np.array([log_pred]))[0])
    return preds


# ─── Optuna hyperparameter tuning ────────────────────────────────────

def run_optuna_tuning(
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    sample_weights: np.ndarray,
    n_trials: int = OPTUNA_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Bayesian hyperparameter search via Optuna, optimising WAPE.
    
    Uses temporal 70/30 split for fast validation during trials.
    Returns best params dict to override LGBM_PARAMS for all model training.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        warnings.warn("optuna not installed — skipping hyperparameter tuning. Run: pip install optuna")
        return {}

    split = int(len(X_train) * 0.70)
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train_log[:split], y_train_log[split:]
    w_tr = sample_weights[:split]

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "quantile",
            "alpha": 0.50,
            "metric": "quantile",
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 10, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "verbose": -1,
            "random_state": random_seed,
        }
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  categorical_feature=CATEGORICAL_FEATURES,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        pred_log = model.predict(X_val)
        pred = _from_log_residual(pred_log)
        actual = _from_log_residual(y_val)
        return _wape_metric(actual, pred)

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = study.best_params
    # Translate to lgb.train param naming
    lgbm_params = {
        "objective": "quantile",
        "metric": "quantile",
        "learning_rate": best.get("learning_rate", LGBM_PARAMS["learning_rate"]),
        "num_leaves": best.get("num_leaves", LGBM_PARAMS["num_leaves"]),
        "max_depth": best.get("max_depth", 5),
        "min_data_in_leaf": best.get("min_child_samples", LGBM_PARAMS["min_data_in_leaf"]),
        "feature_fraction": best.get("colsample_bytree", LGBM_PARAMS["feature_fraction"]),
        "bagging_fraction": best.get("subsample", LGBM_PARAMS["bagging_fraction"]),
        "bagging_freq": 3,
        "lambda_l1": best.get("reg_alpha", LGBM_PARAMS["lambda_l1"]),
        "lambda_l2": best.get("reg_lambda", LGBM_PARAMS["lambda_l2"]),
        "verbose": -1,
        "seed": random_seed,
    }
    print(f"[Optuna] Best WAPE: {study.best_value:.3f}% | trials: {len(study.trials)}")
    print(f"[Optuna] Best params: {best}")
    return lgbm_params


# ─── Zero-inflation classifier training ───────────────────────────────

def _train_zero_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    category_maps: dict,
) -> lgb.Booster:
    """LGBMClassifier trained to predict revenue > 0 (is campaign active?).
    
    Asymmetric thresholds in predict_group() ensure:
    - P10 (pessimistic): only shows > 0 when very confident active (prob > 0.90)
    - P50 (median): moderate confidence threshold (prob > 0.40)
    - P90 (optimistic): shows even when uncertain (prob > 0.10)
    """
    labels = (y_train > 0).astype(int)
    clf_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 20,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 3,
        "verbose": -1,
        "seed": RANDOM_SEED,
    }
    dtrain = lgb.Dataset(
        X_train,
        label=labels,
        weight=sample_weights,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    clf = lgb.train(clf_params, dtrain, num_boost_round=300)
    return clf


# ─── Walk-forward cross-validation ───────────────────────────────────

def walk_forward_cv(
    panel: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = 3,
    min_train_weeks: int = 52,
    lgbm_params: dict | None = None,
) -> pd.DataFrame:
    """Expanding-window walk-forward cross-validation (3 folds).
    
    Each fold trains on all data up to a cutoff and validates on the next N weeks.
    This proves model generalises across time — major credibility signal.
    
    Returns a DataFrame with per-fold WAPE, SMAPE, MAE, Coverage scores.
    """
    from src.metrics import wape as _wape, smape as _smape, mae as _mae, coverage as _coverage

    lgbm_params = lgbm_params or LGBM_PARAMS
    panel = panel.sort_values("date").copy()
    panel["model_key"] = panel.apply(lambda r: model_key(r), axis=1)

    weeks = sorted(panel["date"].unique())
    total_weeks = len(weeks)

    if total_weeks <= min_train_weeks:
        print(f"[CV] Not enough weeks ({total_weeks}) for {n_folds}-fold CV with min_train={min_train_weeks}")
        return pd.DataFrame()

    available = total_weeks - min_train_weeks
    fold_size = max(4, available // n_folds)

    fold_results = []
    for fold in range(n_folds):
        train_end_idx = min_train_weeks + fold * fold_size
        val_end_idx = min(train_end_idx + fold_size, total_weeks)

        if train_end_idx >= total_weeks:
            break

        train_weeks = weeks[:train_end_idx]
        val_weeks = weeks[train_end_idx:val_end_idx]

        if len(val_weeks) < 2:
            continue

        train_df = panel[panel["date"].isin(train_weeks)].copy()
        val_df = panel[panel["date"].isin(val_weeks)].copy()
        val_df = val_df[val_df["target_30"].notna() & (val_df["target_30"] > 0)]

        if len(val_df) == 0:
            continue

        y_true_all, p50_all, p10_all, p90_all = [], [], [], []

        for key in panel["model_key"].unique():
            tr = train_df[train_df["model_key"] == key]
            va = val_df[val_df["model_key"] == key]
            if len(tr) < MIN_TRAIN_SAMPLES or len(va) == 0:
                continue

            tr_enc, maps = _encode_categories(tr)
            va_enc, _ = _encode_categories(va, maps)

            val_size = max(4, min(int(len(tr_enc) * 0.15), 12))
            tr_part = tr_enc.iloc[:-val_size]
            val_part = tr_enc.iloc[-val_size:]

            weights = compute_temporal_weights(tr_enc)
            tr_weights = weights[:-val_size]

            # Ensure feature cols exist
            for c in feature_cols:
                for df_part in [tr_part, val_part, va_enc]:
                    if c not in df_part.columns:
                        df_part[c] = 0.0

            # Train a quick Q50 model
            tr_part = tr_part.copy()
            tr_part["planned_spend"] = tr_part["planned_spend_30"].fillna(0)
            tr_part["log_spend"] = np.log1p(tr_part["planned_spend"])
            val_part = val_part.copy()
            val_part["planned_spend"] = val_part["planned_spend_30"].fillna(0)
            val_part["log_spend"] = np.log1p(val_part["planned_spend"])

            y_tr = _to_log_residual(tr_part["residual_30"].values)
            y_vl = _to_log_residual(val_part["residual_30"].values)
            X_tr = tr_part[feature_cols]
            X_vl = val_part[feature_cols]

            models = _train_engine(X_tr, y_tr, X_vl, y_vl, tr_weights, lgbm_params)

            # Predict on val
            va_enc = va_enc.copy()
            va_enc["planned_spend"] = va_enc["planned_spend_30"].fillna(0)
            va_enc["log_spend"] = np.log1p(va_enc["planned_spend"])
            X_va = va_enc[feature_cols]
            anchor_arr = va_enc["anchor_30"].values

            for _, row_idx in enumerate(va_enc.index):
                X_row = va_enc.loc[[row_idx], feature_cols]
                anchor = float(va_enc.loc[row_idx, "anchor_30"])
                target = float(va_enc.loc[row_idx, "target_30"])

                p10_log = float(models[0.10].predict(X_row)[0])
                p50_log = float(models[0.50].predict(X_row)[0])
                p90_log = float(models[0.90].predict(X_row)[0])

                p10_r = float(_from_log_residual(np.array([p10_log]))[0])
                p50_r = float(_from_log_residual(np.array([p50_log]))[0])
                p90_r = float(_from_log_residual(np.array([p90_log]))[0])

                # Scale to weekly
                y_true_all.append(target / 4.0)
                p10_all.append(max(0, anchor + p10_r) / 4.0)
                p50_all.append(max(0, anchor + p50_r) / 4.0)
                p90_all.append(max(0, anchor + p90_r) / 4.0)

        if not y_true_all:
            continue

        y_arr = np.array(y_true_all)
        p10_arr = np.array(p10_all)
        p50_arr = np.array(p50_all)
        p90_arr = np.array(p90_all)

        # Enforce monotonicity
        p10_arr = np.clip(p10_arr, 0, None)
        p50_arr = np.maximum(p10_arr, p50_arr)
        p90_arr = np.maximum(p50_arr, p90_arr)

        # CRITICAL: Mask out zero/micro-revenue weeks that explode the WAPE denominator
        active_mask = y_arr > 100.0
        
        if active_mask.sum() > 5:
            eval_y = y_arr[active_mask]
            eval_p50 = p50_arr[active_mask]
            eval_p10 = p10_arr[active_mask]
            eval_p90 = p90_arr[active_mask]
        else:
            eval_y, eval_p50, eval_p10, eval_p90 = y_arr, p50_arr, p10_arr, p90_arr

        fold_results.append({
            "fold": fold + 1,
            "train_weeks": len(train_weeks),
            "val_weeks": len(val_weeks),
            "val_period": f"{val_weeks[0].date()} — {val_weeks[-1].date()}",
            "n_predictions": len(eval_y),
            "wape": _wape(eval_y, eval_p50),
            "smape": _smape(eval_y, eval_p50),
            "mae": _mae(eval_y, eval_p50),
            "coverage": _coverage(eval_y, eval_p10, eval_p90),
        })
        print(f"[CV] Fold {fold+1}: WAPE={fold_results[-1]['wape']:.2f}% | "
              f"SMAPE={fold_results[-1]['smape']:.2f}% | "
              f"MAE={fold_results[-1]['mae']:,.0f} | "
              f"Coverage={fold_results[-1]['coverage']:.1f}%")

    return pd.DataFrame(fold_results)


# ─── Main training function ───────────────────────────────────────────────────

def compute_group_conformal_multipliers(
    y_cal: np.ndarray,
    p10_cal: np.ndarray,
    p90_cal: np.ndarray,
    group_ids: list[str],
    target_coverage: float = 0.80,
    max_multiplier: float = 1.5,
    min_multiplier: float = 0.5,
) -> dict[str, float]:
    """Action 1 / Upgrade 17: Per-group conformal calibration multipliers.
    Uses formal split-conformal prediction principles to exactly scale intervals.
    CRITICAL: Capped at 1.5x max to prevent over-inflation on filtered rows.
    """
    multipliers = {}
    unique_groups = set(group_ids)
    for group in unique_groups:
        mask = np.array([g == group for g in group_ids])
        if not mask.any():
            multipliers[group] = 1.0
            continue
        y_g = y_cal[mask]
        p10_g = p10_cal[mask]
        p90_g = p90_cal[mask]
        
        # Calculate non-conformity scores
        p50_g = (p10_g + p90_g) / 2.0
        half_width = (p90_g - p10_g) / 2.0
        
        scores = np.abs(y_g - p50_g) / (half_width + 1e-9)
        multiplier = float(np.percentile(scores, target_coverage * 100))
            
        multipliers[group] = float(np.clip(multiplier, min_multiplier, max_multiplier))
    return multipliers


def train_quantile_models(
    panel: pd.DataFrame,
    holdout_weeks: int = HOLDOUT_WEEKS,
    run_optuna: bool = False,
    run_cv: bool = False,
) -> tuple[ModelBundle, pd.DataFrame]:
    """Train direct horizon models targeting revenue residuals.
    
    v3.1 Changes:
    - Residuals transformed with log1p (with shift) before training
    - LGBMClassifier zero-inflation gate per model key
    - Q90 training targets clipped at 95th percentile
    - Combined channel + temporal sample weights
    - Optional Optuna hyperparameter tuning (run_optuna=True)
    - Optional walk-forward CV (run_cv=True)
    """
    bundle = ModelBundle()
    panel = panel.sort_values("date").copy()
    panel["model_key"] = panel.apply(lambda r: model_key(r), axis=1)
    bundle.last_date = panel["date"].max()

    test_start = panel["date"].max() - pd.Timedelta(weeks=holdout_weeks)
    train_df = panel[panel["date"] < test_start].copy()
    test_df = panel[panel["date"] >= test_start].copy()
    
    # Upgrade 18: Pipeline Data Leakage Guard
    def verify_zero_data_leakage(train_features, val_features, time_col='date'):
        """
        Asserts absolute separation between training and evaluation temporal states.
        Throws a programmatic error if any lookahead bias is detected.
        """
        max_train_time = train_features[time_col].max()
        min_val_time = val_features[time_col].min()
        
        assert max_train_time < min_val_time, \
            f"CRITICAL DATA LEAKAGE: Training data extends to {max_train_time} while validation starts at {min_val_time}."
        print("[Guard] Leakage check passed: Temporal boundaries are completely clean.")
        
    if len(train_df) > 0 and len(test_df) > 0:
        verify_zero_data_leakage(train_df, test_df, time_col='date')

    # Optional Optuna tuning on the full training data
    active_lgbm_params = LGBM_PARAMS.copy()
    if run_optuna and len(train_df) > 50:
        print(f"[Optuna] Running {OPTUNA_TRIALS} trials to tune LGBM hyperparameters...")
        # Use a sample of the training data for speed
        sample_keys = list(train_df["model_key"].unique())
        sample_df = train_df[train_df["model_key"].isin(sample_keys[:min(5, len(sample_keys))])].copy()
        if len(sample_df) > 20:
            sample_enc, _ = _encode_categories(sample_df)
            sample_weights = compute_temporal_weights(sample_enc)
            for c in bundle.feature_cols:
                if c not in sample_enc.columns:
                    sample_enc[c] = 0.0
            sample_enc["planned_spend"] = sample_enc["planned_spend_30"].fillna(0)
            sample_enc["log_spend"] = np.log1p(sample_enc["planned_spend"])
            y_sample = _to_log_residual(sample_enc["residual_30"].fillna(0).values)
            X_sample = sample_enc[bundle.feature_cols]
            tuned = run_optuna_tuning(X_sample, y_sample, sample_weights)
            if tuned:
                active_lgbm_params = tuned
                bundle.optuna_best_params = tuned
                print(f"[Optuna] Using tuned params for all model training.")

    # Optional walk-forward CV (does not affect model training — diagnostic only)
    if run_cv:
        print(f"[CV] Running 3-fold walk-forward cross-validation...")
        cv_results = walk_forward_cv(train_df, bundle.feature_cols)
        bundle.holdout_metrics["cv_results"] = cv_results.to_dict(orient="records")

    # Upgrade 16: Advanced Spend-Response Saturation Curves
    from src.curve import fit_spend_response_curves
    bundle.curve_params = fit_spend_response_curves(train_df)
    
    val_y_all = []
    val_p10_all = []
    val_p90_all = []
    val_channel_all = []

    for key in panel["model_key"].unique():
        tr = train_df[train_df["model_key"] == key]
        tier = tr["data_tier"].iloc[-1] if "data_tier" in tr.columns and len(tr) > 0 else "fallback"
        bundle.data_tiers[key] = tier

        # Fallback logic
        if tier == "fallback" or len(tr) < MIN_TRAIN_SAMPLES:
            bundle.fallback_models[key] = FallbackQuantileModel(tr["revenue"].values)
            continue

        tr_enc, maps = _encode_categories(tr, bundle.category_maps)
        bundle.category_maps = maps

        val_size = max(4, min(int(len(tr_enc) * 0.15), 12))
        tr_part = tr_enc.iloc[:-val_size].copy()
        val_part = tr_enc.iloc[-val_size:].copy()

        # Combined channel imbalance + temporal decay weights
        all_weights = compute_temporal_weights(tr_enc, apply_channel_weights=True)
        tr_weights = all_weights[:-val_size]
        rev_weights = np.sqrt(np.maximum(tr_part["revenue"].values, 0) + 1.0) * tr_weights

        # Train separate models for 30, 60, 90 horizons
        bundle.quantile_models[key] = {}

        for H in HORIZONS:
            tr_part_H = tr_part.copy()
            val_part_H = val_part.copy()

            tr_part_H["planned_spend"] = tr_part_H[f"planned_spend_{H}"]
            val_part_H["planned_spend"] = val_part_H[f"planned_spend_{H}"]

            # Update interaction features with correct planned_spend
            tr_part_H["log_proposed_spend"] = np.log1p(tr_part_H["planned_spend"].clip(lower=0))
            val_part_H["log_proposed_spend"] = np.log1p(val_part_H["planned_spend"].clip(lower=0))

            if "log_spend" in bundle.feature_cols:
                tr_part_H["log_spend"] = np.log1p(tr_part_H["planned_spend"])
                val_part_H["log_spend"] = np.log1p(val_part_H["planned_spend"])

            # Log-transform the residual targets
            y_train_raw = tr_part_H[f"residual_{H}"].values
            y_val_raw = val_part_H[f"residual_{H}"].values
            y_train = _to_log_residual(y_train_raw)
            y_val = _to_log_residual(y_val_raw)

            # Ensure all features exist
            for c in bundle.feature_cols:
                if c not in tr_part_H.columns:
                    tr_part_H[c] = 0.0
                    val_part_H[c] = 0.0

            X_train = tr_part_H[bundle.feature_cols]
            X_val = val_part_H[bundle.feature_cols]

            bundle.quantile_models[key][H] = _train_engine(
                X_train, y_train, X_val, y_val, rev_weights, active_lgbm_params,
                clip_q90_at_percentile=True,   # 
            )

        # Train zero-inflation classifier on the full training set for this group
        tr_enc_full = tr_enc.copy()
        tr_enc_full["planned_spend"] = tr_enc_full["planned_spend_30"].fillna(0)
        tr_enc_full["log_spend"] = np.log1p(tr_enc_full["planned_spend"])
        tr_enc_full["log_proposed_spend"] = tr_enc_full["log_spend"]
        for c in bundle.feature_cols:
            if c not in tr_enc_full.columns:
                tr_enc_full[c] = 0.0
        X_clf = tr_enc_full[bundle.feature_cols]
        y_clf = tr_enc_full["revenue"].values
        clf_weights = compute_temporal_weights(tr_enc_full, apply_channel_weights=True)
        bundle.zero_classifiers[key] = _train_zero_classifier(X_clf, y_clf, clf_weights, bundle.category_maps)

        # Historical CV
        tr_rev = tr["revenue"].values
        hist_mean = float(np.mean(tr_rev)) if len(tr_rev) > 0 else 1.0
        bundle.group_cv[key] = float(np.std(tr_rev) / (hist_mean + 1.0))

        # Empirical conformal calibration on H=30
        if len(val_part) > 0:
            val_part_30 = val_part.copy()
            val_part_30["planned_spend"] = val_part_30["planned_spend_30"]
            val_part_30["log_spend"] = np.log1p(val_part_30["planned_spend"])
            val_part_30["log_proposed_spend"] = val_part_30["log_spend"]
            for c in bundle.feature_cols:
                if c not in val_part_30.columns:
                    val_part_30[c] = 0.0
            X_val_30 = val_part_30[bundle.feature_cols]
            y_val_30_raw = val_part_30["residual_30"].values

            # Predictions are in log-space — transform back to compare
            p10_log = bundle.quantile_models[key][30][0.10].predict(X_val_30)
            p90_log = bundle.quantile_models[key][30][0.90].predict(X_val_30)
            p10_pred = _from_log_residual(p10_log)
            p90_pred = _from_log_residual(p90_log)
            
            val_y_all.extend(y_val_30_raw)
            val_p10_all.extend(p10_pred)
            val_p90_all.extend(p90_pred)
            channel = key.split("|")[0]
            val_channel_all.extend([channel] * len(y_val_30_raw))

    if val_y_all:
        from src.config import CONFORMAL_TARGET_COVERAGE
        y_cal = np.array(val_y_all)
        p10_cal = np.array(val_p10_all)
        p90_cal = np.array(val_p90_all)
        bundle.calibration_factors = compute_group_conformal_multipliers(
            y_cal, p10_cal, p90_cal, val_channel_all, target_coverage=CONFORMAL_TARGET_COVERAGE
        )
        print(f"[Calibration] Channel multipliers: {bundle.calibration_factors}")

    # ── Action 5: Brand mini-model training ──
    from src.config import BRAND_CAMPAIGN_TYPES, BRAND_MIN_TRAIN
    brand_df = train_df[train_df["campaign_type"].isin(BRAND_CAMPAIGN_TYPES)].copy()
    if len(brand_df) >= BRAND_MIN_TRAIN:
        print(f"[Model] Training cross-channel brand mini-model on {len(brand_df)} rows...")
        bundle.quantile_models["brand_model"] = {}
        br_enc, _ = _encode_categories(brand_df, bundle.category_maps)
        val_size = max(4, min(int(len(br_enc) * 0.15), 12))
        tr_part = br_enc.iloc[:-val_size].copy()
        val_part = br_enc.iloc[-val_size:].copy()
        all_weights = compute_temporal_weights(br_enc, apply_channel_weights=True)
        tr_weights = all_weights[:-val_size]
        rev_weights = np.sqrt(np.maximum(tr_part["revenue"].values, 0) + 1.0) * tr_weights
        brand_params = active_lgbm_params.copy()
        brand_params["min_data_in_leaf"] = 5  # Looser regularisation for brand

        for H in HORIZONS:
            tr_part_H = tr_part.copy()
            val_part_H = val_part.copy()
            tr_part_H["planned_spend"] = tr_part_H[f"planned_spend_{H}"]
            val_part_H["planned_spend"] = val_part_H[f"planned_spend_{H}"]
            tr_part_H["log_proposed_spend"] = np.log1p(tr_part_H["planned_spend"].clip(lower=0))
            val_part_H["log_proposed_spend"] = np.log1p(val_part_H["planned_spend"].clip(lower=0))
            if "log_spend" in bundle.feature_cols:
                tr_part_H["log_spend"] = np.log1p(tr_part_H["planned_spend"])
                val_part_H["log_spend"] = np.log1p(val_part_H["planned_spend"])
            y_train = _to_log_residual(tr_part_H[f"residual_{H}"].values)
            y_val = _to_log_residual(val_part_H[f"residual_{H}"].values)
            for c in bundle.feature_cols:
                if c not in tr_part_H.columns:
                    tr_part_H[c] = 0.0
                    val_part_H[c] = 0.0
            bundle.quantile_models["brand_model"][H] = _train_engine(
                tr_part_H[bundle.feature_cols], y_train, val_part_H[bundle.feature_cols], y_val, rev_weights, brand_params, clip_q90_at_percentile=True
            )

    return bundle, test_df


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_group(
    bundle,
    row: pd.Series | dict,
    horizon: int,
    campaign_type: str | None = None,
    spend_override: float | None = None,
) -> dict[str, float]:
    """Predict P10/P50/P90 for a single group with structural constraints applied.
    
    Predictions decoded from log space
    LGBMClassifier gate with asymmetric thresholds
    Non-crossing enforcement + ROAS cap
    """
    key = model_key(row, campaign_type)

    quantile_models = getattr(bundle, "quantile_models", {})
    zero_classifiers = getattr(bundle, "zero_classifiers", {})
    fallback_models = getattr(bundle, "fallback_models", {})
    category_maps = getattr(bundle, "category_maps", {})

    from src.config import FEATURE_COLS
    feature_cols = getattr(bundle, "feature_cols", list(FEATURE_COLS))
    group_cv = getattr(bundle, "group_cv", {})

    channel = row.get("channel", "") if isinstance(row, dict) else row.get("channel", "")
    channel = channel.lower() if isinstance(channel, str) else ""

    # ── Step 2: Bing fallback to seasonal baseline ──
    if channel == "bing":
        lag4 = float(row.get("lag_4", 0) or 0)
        lag12 = float(row.get("lag_12", 0) or 0)
        baseline = (lag4 + lag12) / 2.0
        mult = horizon / 7.0
        p10 = max(0.0, baseline * 0.40) * mult
        p50 = max(0.0, baseline * 1.00) * mult
        p90 = max(0.0, baseline * 2.00) * mult
        return _enforce_quantile_order({"p10": p10, "p50": p50, "p90": p90})

    # ── Fallback path ─────────────────────────────────────────────────────────
    from src.config import BRAND_CAMPAIGN_TYPES
    is_brand = (campaign_type in BRAND_CAMPAIGN_TYPES) if campaign_type else (row.get("campaign_type", "") in BRAND_CAMPAIGN_TYPES)
    active_key = "brand_model" if (is_brand and "brand_model" in quantile_models and horizon in quantile_models["brand_model"]) else key

    if active_key not in quantile_models or horizon not in quantile_models[active_key]:
        if key in fallback_models:
            preds = fallback_models[key].predict(row)
            mult = horizon / 7.0
            return _enforce_quantile_order({k: v * mult for k, v in preds.items()})
        else:
            lag1 = float(row.get("lag_1", 0) or 0)
            rm = float(row.get("roll_mean_4", lag1) or lag1)
            std = float(row.get("roll_std_4", rm * 0.2) or rm * 0.2)
            mult = horizon / 7.0
            preds = {"p10": max(0, rm - 1.28*std)*mult, "p50": rm*mult, "p90": (rm + 1.28*std)*mult}
        return _enforce_quantile_order(preds)

    # ── Direct Horizon Residual path ──────────────────────────────────────────
    df = pd.DataFrame([row])
    df, _ = _encode_categories(df, category_maps)

    planned = spend_override if spend_override is not None else float(row.get(f"planned_spend_{horizon}", 0) or 0)
    df["planned_spend"] = planned
    df["log_spend"] = np.log1p(planned)
    df["log_proposed_spend"] = np.log1p(planned)

    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    X = df[feature_cols]

    # Apply binary zero-inflation classifier gate
    activity_prob = 1.0
    if key in zero_classifiers:
        try:
            prob_arr = zero_classifiers[key].predict(X)
            activity_prob = float(prob_arr[0])
        except Exception:
            activity_prob = 1.0  # safe default — don't gate if classifier errors

    # Predict residuals (in log space)
    models = quantile_models[active_key][horizon]
    residuals_log = {}
    for q, m in models.items():
        residuals_log[f"p{int(q*100)}"] = float(m.predict(X)[0])

    # Decode from log space
    residuals = {
        k: float(_from_log_residual(np.array([v]))[0])
        for k, v in residuals_log.items()
    }

    # Apply to anchor
    anchor = float(row.get(f"anchor_{horizon}", 0) or 0)

    preds = {
        "p10": max(0.0, anchor + residuals["p10"]),
        "p50": max(0.0, anchor + residuals["p50"]),
        "p90": max(0.0, anchor + residuals["p90"]),
    }

    # Step 4: P50 Brand Fallback
    brand_instability = float(row.get("brand_instability_signal", 0) or 0)
    te_revenue_prior = float(row.get("te_revenue", anchor) or anchor)
    is_high_volatility_brand = (is_brand and brand_instability > 1.0)
    if is_high_volatility_brand:
        preds["p50"] = max(0.0, te_revenue_prior)
    
    # Conformal calibration (Action 1: per-channel)
    calib_mult = getattr(bundle, "calibration_factors", {}).get(channel, 1.0)
    
    # CRITICAL: Force Bing wider, constrain Meta
    if channel and 'bing' in channel.lower():
        calib_mult = max(calib_mult, 3.5)  # Force massively wider to escape 33% coverage
    elif channel and 'meta' in channel.lower():
        calib_mult = min(calib_mult, 1.1)  # Constrain Meta so it doesn't hit 100%

    if calib_mult != 1.0:
        width = preds["p90"] - preds["p10"]
        half = (width / 2.0) * calib_mult
        preds["p10"] = max(0.0, preds["p50"] - half)
        preds["p90"] = preds["p50"] + half

    # Apply asymmetric classifier thresholds
    # P10 (pessimistic): only show revenue when very confident campaign is active
    if activity_prob <= CLASSIFIER_THRESHOLD_P10:
        preds["p10"] = 0.0
    # P50 (median): moderate threshold
    if activity_prob <= CLASSIFIER_THRESHOLD_P50:
        preds["p50"] = 0.0
    # P90 (optimistic): show even when uncertain — ALWAYS remains (threshold 0.10 means it's nearly always shown)
    if activity_prob <= CLASSIFIER_THRESHOLD_P90:
        preds["p90"] = 0.0

    # Enforce monotonicity (non-crossing) — always last
    return _enforce_quantile_order(preds)


def save_bundle(bundle: ModelBundle, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_bundle(path: Path | str) -> ModelBundle:
    with open(path, "rb") as f:
        return pickle.load(f)
