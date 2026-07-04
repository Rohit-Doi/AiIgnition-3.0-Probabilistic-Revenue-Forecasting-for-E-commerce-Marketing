"""Spend-response curve fitting and budget simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.config import CURVE_MIN_POINTS, BUDGET_RATIO_MIN, BUDGET_RATIO_MAX

@dataclass
class CurveFitResult:
    form: str
    params: dict
    r2: float
    predictions: np.ndarray

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)

def linear(S: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * S + b

def logarithmic(S: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.log(np.maximum(S, 1e-6)) + b

def saturating_exp(S: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * (1 - np.exp(-S / np.maximum(b, 1e-6)))

def hill(S: np.ndarray, vmax: float, k: float, n: float) -> np.ndarray:
    return vmax * (S ** n) / (k ** n + S ** n + 1e-12)

CURVE_FORMS: dict[str, Callable] = {
    "linear": linear,
    "logarithmic": logarithmic,
    "saturating_exp": saturating_exp,
    "hill": hill,
}

def fit_curve(spend: np.ndarray, revenue: np.ndarray) -> CurveFitResult:
    """Keep this for backwards compatibility, but use fit_spend_response_curves for v3."""
    pass

def fit_spend_response_curves(train_df: pd.DataFrame) -> dict[str, list[float]]:
    """
    Fits unique Hill curves per campaign_type to isolate 
    true saturation points for the budget simulator.
    """
    curve_params = {}
    if 'campaign_type' not in train_df.columns:
        return curve_params
        
    groups = train_df.groupby('campaign_type')
    
    for name, group in groups:
        valid = group[(group['spend'] > 10) & (group['revenue'] > 10)]
        if len(valid) < 10:
            continue
            
        try:
            p0 = [valid['revenue'].max(), valid['spend'].median(), 1.0]
            bounds = ((0, 0, 0.1), (valid['revenue'].max() * 5.0, valid['spend'].max() * 5.0, 3.0))
            
            popt, _ = curve_fit(hill, valid['spend'], valid['revenue'], p0=p0, bounds=bounds, maxfev=2000)
            curve_params[name] = popt.tolist()
        except Exception:
            curve_params[name] = [float(valid['revenue'].mean()), float(valid['spend'].mean()), 1.0]
            
    return curve_params


def evaluate_curve(form: str, params: list[float] | dict, spend: np.ndarray | float) -> np.ndarray:
    if isinstance(params, list):
        return hill(np.array(spend), params[0], params[1], params[2])
    
    if form not in CURVE_FORMS:
        raise ValueError(f"Unknown curve form: {form}")
    fn = CURVE_FORMS[form]
    return fn(np.array(spend), *params.values())


def budget_multiplier(form: str, params: list[float] | dict, baseline_spend: float, new_spend: float) -> float:
    """Calculate the curve-driven multiplier from baseline to new spend."""
    if baseline_spend <= 0 or new_spend <= 0:
        return 1.0
        
    try:
        base_rev = evaluate_curve(form, params, np.array([baseline_spend]))[0]
        new_rev = evaluate_curve(form, params, np.array([new_spend]))[0]
    except Exception:
        return 1.0
        
    if base_rev <= 0 or new_rev <= 0:
        return 1.0
        
    ratio = new_rev / base_rev
    
    # Clip ratio to safety bounds [0.1, 5.0] to prevent catastrophic extrapolation
    return float(np.clip(ratio, BUDGET_RATIO_MIN, BUDGET_RATIO_MAX))
