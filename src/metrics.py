"""Backtest metrics: WAPE (primary), SMAPE, MAPE, MAE, RMSE, pinball loss, coverage, baseline comparison.

Fix 4 (v3.1): WAPE (Weighted Absolute Percentage Error) is now the PRIMARY metric.
- WAPE weights errors by actual revenue size → large-revenue weeks dominate correctly.
- SMAPE remains secondary for cross-comparison with friend's submission.
- Added full_scorecard() convenience function matching the improvement plan spec.
- compare_to_baselines() now ranks by WAPE (not SMAPE).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Core scalar metrics ─────────────────────────────────────────────────────

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error — PRIMARY METRIC (Fix 4).
    
    WAPE = sum(|actual - pred|) / sum(|actual|) × 100
    Weights errors by revenue magnitude: large-revenue weeks dominate.
    Superior to SMAPE for ecommerce where a few peak weeks drive most revenue.
    """
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — secondary metric."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mape(y_true: np.ndarray, y_pred: np.ndarray, min_actual: float = 1.0) -> float:
    mask = y_true >= min_actual
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Alias for wape() — backwards compatibility."""
    return wape(y_true, y_pred)


def wsmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Symmetric MAPE."""
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum((np.abs(y_true) + np.abs(y_pred)) / 2)
    return float((numerator / denominator) * 100) if denominator > 0 else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y_true: np.ndarray, p_lower: np.ndarray, p_upper: np.ndarray) -> float:
    inside = (y_true >= p_lower) & (y_true <= p_upper)
    return float(np.mean(inside) * 100)


def interval_width(p_lower: np.ndarray, p_upper: np.ndarray) -> float:
    return float(np.mean(p_upper - p_lower))


# ─── Full scorecard (Fix 4) ───────────────────────────────────────────────────

def full_scorecard(
    y_true: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    spend: np.ndarray | None = None,
    roas_cap: float = 15.0,
) -> dict:
    """Compute the complete metric suite matching the improvement plan spec.
    
    WAPE is returned first and flagged as PRIMARY.
    """
    active_mask = (p50 > 100) | (y_true > 100)
    
    result = {
        "WAPE": wape(y_true, p50),            # PRIMARY — Fix 4
        "SMAPE": smape(y_true, p50),           # Secondary
        "SMAPE_active": smape(y_true[active_mask], p50[active_mask]) if active_mask.any() else float("nan"),
        "MAE": mae(y_true, p50),
        "RMSE": rmse(y_true, p50),
        "Median_AE": median_ae(y_true, p50),
        "Coverage_P10_P90": coverage(y_true, p10, p90),
        "Interval_Width_pct": (
            np.mean(p90 - p10) / (np.mean(y_true) + 1e-9) * 100
        ),
        "Pinball_Q10": pinball_loss(y_true, p10, 0.10),
        "Pinball_Q50": pinball_loss(y_true, p50, 0.50),
        "Pinball_Q90": pinball_loss(y_true, p90, 0.90),
    }
    if spend is not None:
        result.update(roas_metrics(y_true, p50, spend, roas_cap=roas_cap))
    return result


# ─── ROAS metrics ────────────────────────────────────────────────────────────

def generate_calibration_curve(
    y_true: np.ndarray, preds_dict: dict[float, np.ndarray]
) -> list[dict]:
    """Calculate empirical coverage for various nominal quantiles."""
    points = []
    for q, preds in preds_dict.items():
        empirical = float(np.mean(y_true <= preds) * 100)
        points.append({"nominal_q": q * 100, "empirical_q": empirical})
    return points


def roas_metrics(
    y_revenue: np.ndarray,
    y_pred: np.ndarray,
    spend: np.ndarray,
    spend_min: float = 0.0,
    roas_cap: float = 15.0,   # Fix 14: reduced from 50
) -> dict[str, float]:
    mask = spend > spend_min
    if not mask.any():
        return {
            "mape_roas": float("nan"),
            "wmape_roas": float("nan"),
            "mae_roas": float("nan"),
            "rmse_roas": float("nan"),
            "median_ae_roas": float("nan"),
        }
    actual_roas = np.clip(y_revenue[mask] / spend[mask], 0, roas_cap)
    pred_roas = np.clip(y_pred[mask] / spend[mask], 0, roas_cap)
    roas_mask = actual_roas > 0.01
    
    return {
        "mape_roas": mape(actual_roas[roas_mask], pred_roas[roas_mask], min_actual=0.01) if roas_mask.any() else float("nan"),
        "wmape_roas": wape(actual_roas, pred_roas),
        "mae_roas": mae(actual_roas, pred_roas),
        "rmse_roas": rmse(actual_roas, pred_roas),
        "median_ae_roas": median_ae(actual_roas, pred_roas),
    }


# ─── Threshold masking ────────────────────────────────────────────────────────

def _mask_by_thresholds(
    y_true: np.ndarray,
    spend: np.ndarray | None = None,
    revenue_min: float | None = None,
    spend_min: float | None = None,
) -> np.ndarray:
    mask = np.ones(len(y_true), dtype=bool)
    if revenue_min is not None:
        mask &= y_true >= revenue_min
    if spend is not None and spend_min is not None:
        mask &= spend >= spend_min
    return mask


# ─── Evaluate predictions ─────────────────────────────────────────────────────

def evaluate_predictions(
    y_true: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    p25: np.ndarray | None = None,
    p75: np.ndarray | None = None,
    spend: np.ndarray | None = None,
    revenue_min: float | None = None,
    spend_min: float | None = None,
    roas_cap: float = 15.0,   # Fix 14: reduced from 50
) -> dict:
    active_mask = (p50 > 100) | (y_true > 100)
    
    # WAPE is now the primary metric (Fix 4) — listed first
    results = {
        "wape_p50": wape(y_true, p50),         # PRIMARY (Fix 4)
        "smape_p50": smape(y_true, p50),
        "smape_active_p50": smape(y_true[active_mask], p50[active_mask]) if active_mask.any() else float("nan"),
        "mape_p50": mape(y_true, p50),
        "wmape_p50": wape(y_true, p50),         # alias for backwards compat
        "wsmape_p50": wsmape(y_true, p50),
        "mae_p50": mae(y_true, p50),
        "rmse_p50": rmse(y_true, p50),
        "median_ae_p50": median_ae(y_true, p50),
        "coverage_p10_p90": coverage(y_true, p10, p90),
        "avg_interval_width": interval_width(p10, p90),
    }
    
    if p25 is not None and p75 is not None:
        results["coverage_p25_p75"] = coverage(y_true, p25, p75)

    for q, arr in [(0.10, p10), (0.50, p50), (0.90, p90)]:
        results[f"pinball_q{int(q * 100)}"] = pinball_loss(y_true, arr, q)
        
    if spend is not None:
        results.update(roas_metrics(y_true, p50, spend, spend_min=spend_min or 0.0, roas_cap=roas_cap))

    if revenue_min is not None or spend_min is not None:
        mask = _mask_by_thresholds(y_true, spend=spend, revenue_min=revenue_min, spend_min=spend_min)
        if mask.any():
            results.update(
                {
                    "filtered_n": int(mask.sum()),
                    "filtered_wape_p50": wape(y_true[mask], p50[mask]),    # PRIMARY filtered
                    "filtered_smape_p50": smape(y_true[mask], p50[mask]),
                    "filtered_smape_active_p50": smape(y_true[mask & active_mask], p50[mask & active_mask]) if (mask & active_mask).any() else float("nan"),
                    "filtered_mape_p50": mape(y_true[mask], p50[mask]),
                    "filtered_wmape_p50": wape(y_true[mask], p50[mask]),
                    "filtered_wsmape_p50": wsmape(y_true[mask], p50[mask]),
                    "filtered_mae_p50": mae(y_true[mask], p50[mask]),
                    "filtered_rmse_p50": rmse(y_true[mask], p50[mask]),
                    "filtered_median_ae_p50": median_ae(y_true[mask], p50[mask]),
                    "filtered_coverage_p10_p90": coverage(y_true[mask], p10[mask], p90[mask]),
                    "filtered_avg_interval_width": interval_width(p10[mask], p90[mask]),
                }
            )
            if p25 is not None and p75 is not None:
                results["filtered_coverage_p25_p75"] = coverage(y_true[mask], p25[mask], p75[mask])
                
            for q, arr in [(0.10, p10), (0.50, p50), (0.90, p90)]:
                results[f"filtered_pinball_q{int(q * 100)}"] = pinball_loss(y_true[mask], arr[mask], q)
                
            if spend is not None:
                results.update(
                    {
                        f"filtered_{k}": v
                        for k, v in roas_metrics(
                            y_true[mask],
                            p50[mask],
                            spend[mask],
                            spend_min=spend_min or 0.0,
                            roas_cap=roas_cap,
                        ).items()
                    }
                )
    return results


# ─── Baselines ────────────────────────────────────────────────────────────────

def baseline_lag1(row: pd.Series) -> float:
    return float(row.get("lag_1", 0) or 0)


def baseline_rolling_mean(row: pd.Series) -> float:
    return float(row.get("roll_mean_4", 0) or 0)


def baseline_seasonal(row: pd.Series) -> float:
    lag4 = float(row.get("lag_4", 0) or 0)
    lag12 = float(row.get("lag_12", 0) or 0)
    parts = [v for v in (lag4, lag12) if v > 0]
    return float(np.mean(parts)) if parts else baseline_lag1(row)


def compare_to_baselines(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    model_p50: np.ndarray,
    spend: np.ndarray,
    revenue_min: float | None = None,
    spend_min: float | None = None,
    roas_cap: float = 15.0,   # Fix 14
) -> dict:
    mask = _mask_by_thresholds(y_true, spend=spend, revenue_min=revenue_min, spend_min=spend_min)
    if not mask.any():
        return {"comparison": [], "best_baseline": None, "wape_improvement_vs_best_baseline": 0.0,
                "smape_improvement_vs_best_baseline": 0.0, "lightgbm_beats_baseline": False}

    y_true_eval = y_true[mask]
    model_eval = model_p50[mask]
    spend_eval = spend[mask]
    test_eval = test_df.loc[mask].reset_index(drop=True)

    baselines = {
        "lag_1": np.array([baseline_lag1(row) for _, row in test_eval.iterrows()]),
        "rolling_mean_4": np.array([baseline_rolling_mean(row) for _, row in test_eval.iterrows()]),
        "seasonal_lag4_12": np.array([baseline_seasonal(row) for _, row in test_eval.iterrows()]),
    }
    rows = []
    model_row = {
        "model": "lightgbm_p50",
        "wape": wape(y_true_eval, model_eval),   # PRIMARY (Fix 4)
        "smape": smape(y_true_eval, model_eval),
        "mape": mape(y_true_eval, model_eval),
        "wmape": wape(y_true_eval, model_eval),
        "mae": mae(y_true_eval, model_eval),
        "rmse": rmse(y_true_eval, model_eval),
        "median_ae": median_ae(y_true_eval, model_eval),
    }
    model_row.update(roas_metrics(y_true_eval, model_eval, spend_eval, spend_min=spend_min or 0.0, roas_cap=roas_cap))
    rows.append(model_row)

    for name, preds in baselines.items():
        row = {
            "model": name,
            "wape": wape(y_true_eval, preds),    # PRIMARY (Fix 4)
            "smape": smape(y_true_eval, preds),
            "mape": mape(y_true_eval, preds),
            "wmape": wape(y_true_eval, preds),
            "mae": mae(y_true_eval, preds),
            "rmse": rmse(y_true_eval, preds),
            "median_ae": median_ae(y_true_eval, preds),
        }
        row.update(roas_metrics(y_true_eval, preds, spend_eval, spend_min=spend_min or 0.0, roas_cap=roas_cap))
        rows.append(row)

    # Fix 4: Rank baselines by WAPE (not SMAPE)
    best_baseline = min(rows[1:], key=lambda r: r["wape"])
    wape_improvement = best_baseline["wape"] - model_row["wape"]
    smape_improvement = best_baseline["smape"] - model_row["smape"]
    return {
        "comparison": rows,
        "best_baseline": best_baseline["model"],
        "wape_improvement_vs_best_baseline": wape_improvement,
        "smape_improvement_vs_best_baseline": smape_improvement,
        "lightgbm_beats_baseline": wape_improvement > 0,
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
