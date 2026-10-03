"""Three-stage probabilistic forecaster for 30/60/90-day revenue, spend and ROAS.

Stage 1 - structural median.  A regularised log-linear model predicts the
    log-ratio between the future total and the 28-day run-rate from momentum,
    level-reversion, seasonal-analog and efficiency terms.  It has no drift term:
    a series with no signal is forecast at its run-rate.  It is fitted per horizon
    and per hierarchy level group (aggregates react less to a noisy week than
    single campaigns do).  Because it is linear in logs, each forecast decomposes
    exactly into multiplicative drivers.
Stage 2 - uncertainty shape.  LightGBM quantile models learn the 10th and 90th
    percentile of the stage-1 residual, so interval width and skew depend on the
    series (level, size, volatility, intermittency, season).
Stage 3 - calibration.  Conformal offsets measured on rolling-origin
    out-of-sample residuals widen the intervals until they deliver the nominal
    coverage.

Budget response.  A separate regression measures the elasticity of revenue to
spend (how much of a spend change passes through to revenue).  Scenario forecasts
multiply the baseline by (planned spend / expected spend) ** elasticity.

The trained artifact is a plain dict of floats, strings and lists (LightGBM
boosters are stored as model text), so unpickling does not depend on the exact
versions of any library classes.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    CONFORMAL_DEFAULT_OFFSET,
    CONFORMAL_MIN_ORIGINS,
    CONFORMAL_MIN_SAMPLES,
    ELASTICITY_BOUNDS,
    HORIZONS,
    INTERVAL_COVERAGE,
    MOMENTUM_DEAD_ZONE,
    QGBM_PARAMS,
    QGBM_ROUNDS,
    RANDOM_SEED,
    RIDGE_ALPHA,
    SAMPLE_WEIGHT_POWER,
    SEASONAL_CLIP,
)
from src.features import CATEGORICAL_COLS, FEATURE_COLS

MODEL_VERSION = "4.0"

# ─── Stage-1 design ──────────────────────────────────────────────────────────
# Short-term momentum enters through a dead zone (see design()): mom_x is the
# 7- or 14-day vs 28-day log-ratio with the first MOMENTUM_DEAD_ZONE removed.
LINEAR_FEATURES = {
    "rev": ["mom_r7", "mom_r14", "r56_28", "r91_28", "r182_28", "mom_s7", "mom_s14", "s91_28",
            "roas_dev", "port_r7_28", "parent_r7_28", "zero_rev_share_28",
            "seas_peak", "seas_off", "seas_chan_dev"],
    "sp": ["mom_s7", "mom_s14", "s56_28", "s91_28", "s182_28", "mom_r7", "roas_dev",
           "port_s7_28", "parent_s7_28", "zero_sp_share_28", "seassp_peak", "seassp_off"],
}
# Coefficients are shrunk towards these values instead of towards zero: with no
# evidence to the contrary, a peak-season analog ratio is taken at face value.
LINEAR_PRIOR = {"rev": {"seas_peak": 1.0}, "sp": {"seassp_peak": 1.0}}

# How stage-1 terms are grouped when a forecast is explained.
DRIVER_GROUPS = {
    "momentum": ["mom_r7", "mom_r14", "mom_s7", "mom_s14"],
    "reversion": ["r56_28", "r91_28", "r182_28", "s91_28"],
    "seasonality": ["seas_peak", "seas_off", "seas_chan_dev"],
    "efficiency": ["roas_dev"],
    "context": ["port_r7_28", "parent_r7_28"],
    "structural": ["zero_rev_share_28"],
}

QUANTILE_TARGETS = ["rev", "sp", "roas"]
MIN_HALF_WIDTH = 0.05          # log-units: no interval bound sits closer than ~5% to the median
QUANTILE_FEATURES = FEATURE_COLS + ["z1_rev", "z1_sp"]
LEVEL_GROUPS = {0: "top", 1: "top", 2: "type", 3: "campaign"}          # stage-1 coefficient sets
CALIBRATION_GROUPS = {0: "blended", 1: "channel", 2: "type", 3: "campaign"}   # stage-3 buckets
POOLED = "all"                 # stage-1 spec used when a level group has too few rows
MIN_GROUP_ROWS = 60


def _dead_zone(x: pd.Series, width: float = MOMENTUM_DEAD_ZONE) -> np.ndarray:
    """Soft threshold: moves smaller than `width` are noise and count as zero."""
    v = x.fillna(0.0).to_numpy(dtype=float)
    return np.sign(v) * np.maximum(np.abs(v) - width, 0.0)


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Add baselines, derived linear terms and (when targets exist) log-ratio targets."""
    d = df.copy()
    h = d["horizon"].to_numpy(dtype=float)
    d["B_rev"] = d["base_rev_daily"] * h
    d["C_rev"] = d["c_rev_daily"] * h
    d["B_sp"] = d["base_sp_daily"] * h
    d["C_sp"] = d["c_sp_daily"] * h

    lo, hi = SEASONAL_CLIP
    seas_rev = d["seas_port_rev"].fillna(0.0).clip(lo, hi)
    seas_sp = d["seas_port_sp"].fillna(0.0).clip(lo, hi)
    peaky = (d["peak_share_target"] > 0) | (d["peak_share_base"] > 0)
    d["peaky"] = peaky
    d["no_seasonal"] = d["seas_port_rev"].isna()
    d["seas_peak"] = seas_rev * peaky
    d["seas_off"] = seas_rev * ~peaky
    d["seassp_peak"] = seas_sp * peaky
    d["seassp_off"] = seas_sp * ~peaky
    d["seas_chan_dev"] = (d["seas_chan_rev"] - d["seas_port_rev"]).fillna(0.0).clip(-1.5, 1.5)
    d["roas_dev"] = d["log_roas_28"] - d["log_roas_91"]
    for name, col in (("mom_r7", "r7_28"), ("mom_r14", "r14_28"), ("mom_s7", "s7_28"), ("mom_s14", "s14_28")):
        d[name] = _dead_zone(d[col])
    d["level_group"] = d["level_code"].map(LEVEL_GROUPS).fillna("campaign")
    d["calib_group"] = d["level_code"].map(CALIBRATION_GROUPS).fillna("campaign")

    if "y_rev" in d.columns:
        d["z_rev"] = np.log((d["y_rev"] + d["C_rev"]) / (d["B_rev"] + d["C_rev"]))
        d["z_sp"] = np.log((d["y_sp"] + d["C_sp"]) / (d["B_sp"] + d["C_sp"]))
    return d


def _sample_weights(d: pd.DataFrame) -> np.ndarray:
    """Bigger series carry more weight, sub-linearly, normalised to mean 1."""
    w = np.power(d["base_rev_daily"].to_numpy() + d["c_rev_daily"].to_numpy(), SAMPLE_WEIGHT_POWER)
    return w / w.mean()


def _ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float,
           intercept: bool = True) -> tuple[np.ndarray, float]:
    """Weighted ridge regression in closed form; the intercept, if any, is unpenalised."""
    if not intercept:
        A = (X * w[:, None]).T @ X + alpha * np.eye(X.shape[1])
        return np.linalg.solve(A, (X * w[:, None]).T @ y), 0.0
    sw = w.sum()
    x_mean = (X * w[:, None]).sum(axis=0) / sw
    y_mean = float((y * w).sum() / sw)
    Xc, yc = X - x_mean, y - y_mean
    A = (Xc * w[:, None]).T @ Xc + alpha * np.eye(X.shape[1])
    coef = np.linalg.solve(A, (Xc * w[:, None]).T @ yc)
    return coef, y_mean - float(x_mean @ coef)


def _fit_linear(d: pd.DataFrame, feats: list[str], target: str, prior: dict[str, float],
                alpha: float = RIDGE_ALPHA) -> dict:
    """Stage-1 coefficients for one horizon: one spec per level group plus a pooled one.

    No intercept: with every term at zero the forecast is the run-rate itself.
    """
    def one(a: pd.DataFrame) -> dict:
        X = a[feats].fillna(0.0).to_numpy(dtype=float)
        offset = sum(a[k].to_numpy(dtype=float) * v for k, v in prior.items()) if prior else 0.0
        coef, _ = _ridge(X, a[target].to_numpy(dtype=float) - offset, _sample_weights(a), alpha, intercept=False)
        return {f: float(c) + float(prior.get(f, 0.0)) for f, c in zip(feats, coef)}

    specs = {POOLED: one(d)}
    for group in sorted(set(LEVEL_GROUPS.values())):
        a = d[d["level_group"] == group]
        if len(a) >= MIN_GROUP_ROWS:
            specs[group] = one(a)
    return specs


def _terms(d: pd.DataFrame, specs: dict, terms: list[str] | None = None) -> np.ndarray:
    """Sum of coefficient x feature over `terms` (default: all), using each row's group spec."""
    out = np.zeros(len(d))
    groups = d["level_group"].to_numpy()
    for group in np.unique(groups):
        coef = specs.get(group, specs[POOLED])
        m = groups == group
        for f in (terms if terms is not None else coef):
            out[m] += coef[f] * d.loc[m, f].fillna(0.0).to_numpy(dtype=float)
    return out


def _predict_by_horizon(d: pd.DataFrame, by_horizon: dict, terms: list[str] | None = None) -> np.ndarray:
    z = np.zeros(len(d))
    for h, specs in by_horizon.items():
        m = (d["horizon"] == int(h)).to_numpy()
        if m.any():
            z[m] = _terms(d[m], specs, terms)
    return z


# ─── Budget elasticity ───────────────────────────────────────────────────────

def _fit_elasticity(d: pd.DataFrame, alpha: float = RIDGE_ALPHA) -> dict:
    """Pass-through of a spend change to revenue, with channel / type deviations.

    Estimated on window-over-window changes: the revenue log-ratio is regressed
    on the spend log-ratio realised over the same horizon while controlling for
    everything known at the origin.  Deviations are ridge-shrunk towards the
    pooled elasticity, so thin groups borrow strength from the rest.
    """
    base = [f for f in LINEAR_FEATURES["rev"] if f != "seas_peak"]
    X = d[base].fillna(0.0).copy()
    X["z_sp"] = d["z_sp"]
    inter = {}
    for code in sorted(d["channel_code"].unique()):
        if code == 0:
            continue
        name = f"z_sp|channel|{int(code)}"
        X[name] = d["z_sp"] * (d["channel_code"] == code)
        inter[name] = ("channel", int(code))
    for code in sorted(d["type_code"].unique()):
        if code == 0:
            continue
        name = f"z_sp|type|{int(code)}"
        X[name] = d["z_sp"] * (d["type_code"] == code)
        inter[name] = ("type", int(code))
    coef, _ = _ridge(X.to_numpy(dtype=float), d["z_rev"].to_numpy(dtype=float), _sample_weights(d), alpha)
    by_name = dict(zip(X.columns, coef))
    out = {"base": float(by_name["z_sp"]), "channel": {}, "type": {}}
    for name, (kind, code) in inter.items():
        out[kind][code] = float(by_name[name])
    return out


def elasticity_for(d: pd.DataFrame, specs: dict) -> np.ndarray:
    """Per-row elasticity = pooled + channel deviation + type deviation, bounded."""
    beta = np.zeros(len(d))
    for h, spec in specs.items():
        m = (d["horizon"] == int(h)).to_numpy()
        if not m.any():
            continue
        ch = d.loc[m, "channel_code"].map(lambda c: spec["channel"].get(int(c), 0.0)).to_numpy()
        ty = d.loc[m, "type_code"].map(lambda c: spec["type"].get(int(c), 0.0)).to_numpy()
        beta[m] = spec["base"] + ch + ty
    return np.clip(beta, *ELASTICITY_BOUNDS)


# ─── Stage 2: quantile models on stage-1 residuals ───────────────────────────

def _residual(d: pd.DataFrame, target: str) -> np.ndarray:
    if target == "rev":
        return (d["z_rev"] - d["z1_rev"]).to_numpy()
    if target == "sp":
        return (d["z_sp"] - d["z1_sp"]).to_numpy()
    return ((d["z_rev"] - d["z_sp"]) - (d["z1_rev"] - d["z1_sp"])).to_numpy()


def _fit_quantile_models(d: pd.DataFrame) -> dict:
    lo, hi = (1 - INTERVAL_COVERAGE) / 2, (1 + INTERVAL_COVERAGE) / 2
    w = _sample_weights(d)
    models: dict[str, dict[str, str]] = {}
    for target in QUANTILE_TARGETS:
        y = _residual(d, target)
        models[target] = {}
        for name, q in (("lo", lo), ("hi", hi)):
            params = {**QGBM_PARAMS, "objective": "quantile", "alpha": q}
            ds = lgb.Dataset(d[QUANTILE_FEATURES], label=y, weight=w,
                             categorical_feature=CATEGORICAL_COLS, free_raw_data=False)
            booster = lgb.train(params, ds, num_boost_round=QGBM_ROUNDS)
            models[target][name] = booster.model_to_string()
    return models


_BOOSTER_CACHE: dict[int, lgb.Booster] = {}


def _booster(model_str: str) -> lgb.Booster:
    key = hash(model_str)
    if key not in _BOOSTER_CACHE:
        _BOOSTER_CACHE[key] = lgb.Booster(model_str=model_str)
    return _BOOSTER_CACHE[key]


# ─── Fit / score ─────────────────────────────────────────────────────────────

def fit_core(samples: pd.DataFrame) -> dict:
    """Fit stage 1, the budget elasticity and stage 2 on a training sample table."""
    d = design(samples)
    # Rows that touch the peak season but have no analog year cannot be explained
    # by anything the model is allowed to see; leaving them in would teach the
    # momentum terms to chase a seasonal spike.
    d = d[d["alive"] & ~(d["no_seasonal"] & d["peaky"])].reset_index(drop=True)
    if d.empty:
        raise ValueError("No usable training samples.")

    core: dict = {"linear": {"rev": {}, "sp": {}}, "elasticity": {}}
    for h in sorted(d["horizon"].unique()):
        a = d[d["horizon"] == h]
        core["linear"]["rev"][int(h)] = _fit_linear(a, LINEAR_FEATURES["rev"], "z_rev", LINEAR_PRIOR["rev"])
        core["linear"]["sp"][int(h)] = _fit_linear(a, LINEAR_FEATURES["sp"], "z_sp", LINEAR_PRIOR["sp"])
        core["elasticity"][int(h)] = _fit_elasticity(a)

    d["z1_rev"] = _predict_by_horizon(d, core["linear"]["rev"])
    d["z1_sp"] = _predict_by_horizon(d, core["linear"]["sp"])
    core["quantile_models"] = _fit_quantile_models(d)
    core["n_samples"] = int(len(d))
    return core


def raw_scores(core: dict, feats: pd.DataFrame) -> pd.DataFrame:
    """Stage-1 medians and uncalibrated stage-2 residual quantiles for feature rows."""
    d = design(feats)
    d["z1_rev"] = _predict_by_horizon(d, core["linear"]["rev"])
    d["z1_sp"] = _predict_by_horizon(d, core["linear"]["sp"])
    d["beta"] = elasticity_for(d, core["elasticity"])
    X = d[QUANTILE_FEATURES].to_numpy(dtype=float)
    for target in QUANTILE_TARGETS:
        for name in ("lo", "hi"):
            d[f"q_{target}_{name}"] = _booster(core["quantile_models"][target][name]).predict(X)
    # stage-1 contributions, grouped for explanation (revenue model)
    for group, terms in DRIVER_GROUPS.items():
        d[f"drv_{group}"] = _predict_by_horizon(d, core["linear"]["rev"], terms)
    return d


# ─── Stage 3: conformal calibration ──────────────────────────────────────────

def _bucket(group: str, horizon: int) -> str:
    return f"{group}|{int(horizon)}"


def fit_conformal(oos: pd.DataFrame) -> dict:
    """Offsets that make the stage-2 quantiles deliver nominal coverage out of sample.

    `oos` holds rolling-origin predictions together with realised targets.  For
    every (level group, horizon) bucket the offset is the amount by which each
    bound must move so that only (1 - coverage) / 2 of realised values fall
    outside it.  Residuals issued at the same origin are strongly correlated, so
    a bucket must span enough distinct origins as well as enough rows; a thin
    bucket borrows from blended + channel together (for those two levels), then
    from the pooled horizon, then falls back to a default.
    """
    tail = (1 + INTERVAL_COVERAGE) / 2
    oos = oos[oos["alive"]]
    table: dict[str, dict[str, list[float]]] = {t: {} for t in QUANTILE_TARGETS}

    def offsets(cal: pd.DataFrame, target: str) -> list[float] | None:
        if len(cal) < CONFORMAL_MIN_SAMPLES or cal["origin"].nunique() < CONFORMAL_MIN_ORIGINS:
            return None
        level = min(1.0, tail * (1 + 1 / len(cal)))
        r = _residual(cal, target)
        return [float(np.quantile(cal[f"q_{target}_lo"].to_numpy() - r, level)),
                float(np.quantile(r - cal[f"q_{target}_hi"].to_numpy(), level))]

    for h in sorted(oos["horizon"].unique()):
        pooled = oos[oos["horizon"] == h]
        top = pooled[pooled["level_group"] == "top"]
        for group in CALIBRATION_GROUPS.values():
            cal = pooled[pooled["calib_group"] == group]
            wider = top if group in ("blended", "channel") else pooled
            for target in QUANTILE_TARGETS:
                table[target][_bucket(group, h)] = (
                    offsets(cal, target) or offsets(wider, target) or offsets(pooled, target)
                    or [CONFORMAL_DEFAULT_OFFSET, CONFORMAL_DEFAULT_OFFSET])
    return table


def apply_conformal(scored: pd.DataFrame, conformal: dict) -> pd.DataFrame:
    """Calibrated residual bounds lo_<target> / hi_<target> for revenue, spend and ROAS."""
    d = scored.copy()
    keys = [_bucket(g, h) for g, h in zip(d["calib_group"], d["horizon"])]
    default = [CONFORMAL_DEFAULT_OFFSET, CONFORMAL_DEFAULT_OFFSET]
    for target in QUANTILE_TARGETS:
        off = np.array([conformal.get(target, {}).get(k, default) for k in keys], dtype=float)
        q_lo, q_hi = d[f"q_{target}_lo"].to_numpy(), d[f"q_{target}_hi"].to_numpy()
        # Calibration may narrow a bound, but never past half of what stage 2
        # asked for and never to nothing: a run of one-sided misses in a short
        # calibration window must not collapse the opposite side of the band.
        d[f"lo_{target}"] = np.minimum(np.minimum(q_lo - off[:, 0], 0.5 * q_lo), -MIN_HALF_WIDTH)
        d[f"hi_{target}"] = np.maximum(np.maximum(q_hi + off[:, 1], 0.5 * q_hi), MIN_HALF_WIDTH)
    return d


# ─── Bundle I/O ──────────────────────────────────────────────────────────────

def make_bundle(core: dict, conformal: dict, seasonal_reference: dict, trained_on: dict,
                backtest: dict | None = None) -> dict:
    return {
        "version": MODEL_VERSION,
        "horizons": list(HORIZONS),
        "interval_coverage": INTERVAL_COVERAGE,
        "linear": core["linear"],
        "elasticity": core["elasticity"],
        "quantile_models": core["quantile_models"],
        "quantile_features": list(QUANTILE_FEATURES),
        "conformal": conformal,
        "seasonal_reference": seasonal_reference,
        "trained_on": trained_on,
        "backtest": backtest or {},
        "seed": RANDOM_SEED,
    }


def save_bundle(bundle: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=4)


def load_bundle(path: Path | str) -> dict:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict) or "linear" not in bundle:
        raise ValueError(
            f"{path} is not an AIgnition v{MODEL_VERSION} model bundle. Retrain with: python -m src.train")
    return bundle
