"""Budget planning on top of the baseline forecast.

Everything here works on the forecast table produced by src.forecast and on the
fitted spend elasticities it carries; no model is re-run.  The response of a
campaign type to a different budget is the concave power curve

    revenue(S) = (revenue_0 + c) * ((S + c_s) / (S_0 + c_s)) ** elasticity - c

where S_0 / revenue_0 are the expected spend and revenue.  elasticity < 1 means
diminishing returns: each extra dollar buys less than the one before.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import BUDGET_MULTIPLIER_BOUNDS, EXTRAPOLATION_WARN_RATIO
from src.features import _cumsum
from src.forecast import apply_spend_plan
from src.hierarchy import Hierarchy

_Z90 = 1.2815515655446004      # standard normal 90th percentile


def type_rows(fc: pd.DataFrame, horizon: int, active_only: bool = True) -> pd.DataFrame:
    """Campaign-type rows of one horizon - the level budgets are planned at."""
    t = fc[(fc["level"] == "campaign_type") & (fc["horizon_days"] == horizon)]
    if active_only:
        t = t[t["spend_p50"] > 0]
    return t.reset_index(drop=True)


def _revenue_at(spend: np.ndarray, t: pd.DataFrame) -> np.ndarray:
    ratio = (spend + t["_c_sp"].to_numpy()) / (t["spend_p50"].to_numpy() + t["_c_sp"].to_numpy())
    rev = (t["revenue_p50"].to_numpy() + t["_c_rev"].to_numpy()) * ratio ** t["elasticity"].to_numpy()
    return np.maximum(rev - t["_c_rev"].to_numpy(), 0.0)


def _marginal_at(spend: np.ndarray, t: pd.DataFrame) -> np.ndarray:
    """d revenue / d spend: what the next dollar is expected to return."""
    c_sp = t["_c_sp"].to_numpy()
    ratio = (spend + c_sp) / (t["spend_p50"].to_numpy() + c_sp)
    beta = t["elasticity"].to_numpy()
    return beta * (t["revenue_p50"].to_numpy() + t["_c_rev"].to_numpy()) * ratio ** beta / (spend + c_sp)


def marginal_roas(fc: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Average vs marginal ROAS per campaign type at the current plan."""
    t = type_rows(fc, horizon)
    out = t[["series_id", "channel", "campaign_type", "spend_p50", "revenue_p50", "roas_p50", "elasticity"]].copy()
    out["marginal_roas"] = _marginal_at(t["spend_p50"].to_numpy(), t)
    return out.sort_values("marginal_roas", ascending=False).reset_index(drop=True)


def response_curve(fc: pd.DataFrame, series_id: int, horizon: int, n: int = 41) -> pd.DataFrame:
    """Revenue / ROAS / marginal ROAS for one campaign type across the budget range."""
    t = fc[(fc["series_id"] == series_id) & (fc["horizon_days"] == horizon)].reset_index(drop=True)
    lo, hi = BUDGET_MULTIPLIER_BOUNDS
    mult = np.linspace(lo, hi, n)
    spend = mult * float(t["spend_p50"].iloc[0])
    rep = pd.concat([t] * n, ignore_index=True)
    rev = _revenue_at(spend, rep)
    c_rev = float(t["_c_rev"].iloc[0])
    return pd.DataFrame({
        "multiplier": mult,
        "spend": spend,
        "revenue_p50": rev,
        "revenue_p10": np.maximum((rev + c_rev) * math.exp(float(t["_lo_budget"].iloc[0])) - c_rev, 0.0),
        "revenue_p90": (rev + c_rev) * math.exp(float(t["_hi_budget"].iloc[0])) - c_rev,
        "roas": np.where(spend > 0, rev / np.where(spend > 0, spend, 1.0), 0.0),
        "marginal_roas": _marginal_at(spend, rep),
    })


def scale_plan(fc: pd.DataFrame, multipliers: dict[tuple[str, str], float], default: float = 1.0) -> pd.Series:
    """Spend plan for all horizons from {(channel, type) or (channel, '*'): multiplier}."""
    t = fc[fc["level"] == "campaign_type"]
    mult = np.array([multipliers.get((c, ct), multipliers.get((c, "*"), default))
                     for c, ct in zip(t["channel"], t["campaign_type"])], dtype=float)
    return pd.Series(t["spend_p50"].to_numpy() * mult,
                     index=pd.MultiIndex.from_arrays([t["series_id"], t["horizon_days"]]))


def run_scenario(fc: pd.DataFrame, multipliers: dict[tuple[str, str], float], default: float = 1.0) -> pd.DataFrame:
    return apply_spend_plan(fc, scale_plan(fc, multipliers, default))


def optimise_budget(fc: pd.DataFrame, horizon: int, total_budget: float | None = None,
                    bounds: tuple[float, float] = (0.5, 2.0)) -> pd.DataFrame:
    """Allocation of a fixed budget across campaign types that maximises median revenue.

    Each type may move between bounds[0] x and bounds[1] x its expected spend.
    The objective is concave, so the optimum equalises marginal ROAS across every
    type that is not pinned to a bound.
    """
    t = type_rows(fc, horizon)
    if t.empty:
        return pd.DataFrame()
    s0 = t["spend_p50"].to_numpy()
    lo, hi = s0 * bounds[0], s0 * bounds[1]
    budget = float(s0.sum()) if total_budget is None else float(np.clip(total_budget, lo.sum(), hi.sum()))

    start = np.clip(s0 * budget / s0.sum(), lo, hi)
    scale = max(float(_revenue_at(s0, t).sum()), 1.0)
    res = minimize(
        lambda s: -_revenue_at(s, t).sum() / scale,
        start,
        jac=lambda s: -_marginal_at(s, t) / scale,
        bounds=list(zip(lo, hi)),
        constraints=[{"type": "eq", "fun": lambda s: s.sum() - budget, "jac": lambda s: np.ones_like(s)}],
        method="SLSQP",
        options={"maxiter": 300, "ftol": 1e-10},
    )
    spend = np.clip(res.x, lo, hi) if res.success else start

    out = t[["series_id", "channel", "campaign_type", "elasticity"]].copy()
    out["spend_current"] = s0
    out["spend_recommended"] = spend
    out["change_pct"] = (spend / s0 - 1) * 100
    out["revenue_current"] = _revenue_at(s0, t)
    out["revenue_recommended"] = _revenue_at(spend, t)
    out["marginal_roas_current"] = _marginal_at(s0, t)
    out["marginal_roas_recommended"] = _marginal_at(spend, t)
    out["at_bound"] = np.where(np.isclose(spend, lo, rtol=1e-3), "min",
                               np.where(np.isclose(spend, hi, rtol=1e-3), "max", ""))
    return out.sort_values("spend_recommended", ascending=False).reset_index(drop=True)


def multipliers_from_spend(fc: pd.DataFrame, horizon: int,
                           planned: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Turn planned dollars per (channel, campaign type) for one horizon into budget multipliers.

    Types the plan does not mention keep their expected spend.
    """
    t = type_rows(fc, horizon, active_only=False).set_index(["channel", "campaign_type"])["spend_p50"]
    out = {}
    for key, dollars in planned.items():
        expected = float(t.get(key, 0.0))
        if expected > 0 and dollars is not None and np.isfinite(dollars):
            out[key] = max(float(dollars), 0.0) / expected
    return out


def last_year_multipliers(hier: Hierarchy, fc: pd.DataFrame, horizon: int,
                          shift_days: int = 364) -> dict[tuple[str, str], float]:
    """Budget multipliers that replay what each campaign type spent in the same window a year ago.

    Useful as a starting plan before a seasonal peak.  Empty when the history
    does not reach back far enough.
    """
    a = hier.n_days - shift_days          # first day of the analog window
    b = a + horizon - 1
    if a < 0 or b > hier.n_days - 1:
        return {}
    t = type_rows(fc, horizon)
    lo, hi = BUDGET_MULTIPLIER_BOUNDS
    out = {}
    for _, r in t.iterrows():
        analog = float(hier.spend[int(r["series_id"]), a:b + 1].sum())
        if analog > 0 and r["spend_p50"] > 0:
            out[(r["channel"], r["campaign_type"])] = float(np.clip(analog / r["spend_p50"], lo, hi))
    return out


# ─── Reading a forecast as a probability ─────────────────────────────────────

def probability_at_least(p10: float, p50: float, p90: float, target: float) -> float:
    """P(outcome >= target) from three quantiles, via a two-piece (split) normal.

    Fitted in log space when the lower quantile is positive, so a right-skewed
    forecast keeps its skew; otherwise in levels.
    """
    if not (p90 > p50 > p10 >= 0):
        return float(p50 >= target)
    if target <= 0:
        return 1.0
    use_log = p10 > 0
    f = math.log if use_log else (lambda v: v)
    mid, x = f(p50), f(target)
    sigma = (mid - f(p10)) / _Z90 if x < mid else (f(p90) - mid) / _Z90
    z = (x - mid) / max(sigma, 1e-12)
    return float(0.5 * math.erfc(z / math.sqrt(2)))


def quantile_from_three(p10: float, p50: float, p90: float, q: float) -> float:
    """Any quantile of the same split-normal (used for the forecast cone)."""
    z = NormalDist().inv_cdf(q)
    if not (p90 > p50 > p10 >= 0):
        return p50
    if p10 > 0:
        sigma = (math.log(p50) - math.log(p10)) / _Z90 if z < 0 else (math.log(p90) - math.log(p50)) / _Z90
        return math.exp(math.log(p50) + z * sigma)
    sigma = (p50 - p10) / _Z90 if z < 0 else (p90 - p50) / _Z90
    return max(p50 + z * sigma, 0.0)


def outcome_curve(p10: float, p50: float, p90: float, n: int = 99) -> pd.DataFrame:
    """Value at every percentile 1..99 of the split-normal: the forecast as a full distribution."""
    q = np.linspace(0.01, 0.99, n)
    return pd.DataFrame({"percentile": q * 100,
                         "value": [quantile_from_three(p10, p50, p90, float(x)) for x in q]})


# ─── Guard rails ─────────────────────────────────────────────────────────────

def extrapolation_warnings(hier: Hierarchy, scenario: pd.DataFrame, horizon: int) -> list[dict]:
    """Campaign types whose planned spend exceeds anything they have spent in a window that long."""
    cs = _cumsum(hier.spend)
    out = []
    t = scenario[(scenario["level"] == "campaign_type") & (scenario["horizon_days"] == horizon)]
    t = t[t["spend_p50"] >= 0.01 * t["spend_p50"].sum()]      # immaterial lines are not worth a warning
    for _, r in t.iterrows():
        sid = int(r["series_id"])
        if hier.n_days <= horizon:
            hist_max = float(cs[sid, -1])
        else:
            hist_max = float((cs[sid, horizon:] - cs[sid, :-horizon]).max())
        if hist_max > 0 and r["spend_p50"] > EXTRAPOLATION_WARN_RATIO * hist_max:
            out.append({
                "channel": r["channel"], "campaign_type": r["campaign_type"],
                "planned_spend": float(r["spend_p50"]), "historical_max_spend": hist_max,
                "ratio": float(r["spend_p50"] / hist_max),
            })
    return out
