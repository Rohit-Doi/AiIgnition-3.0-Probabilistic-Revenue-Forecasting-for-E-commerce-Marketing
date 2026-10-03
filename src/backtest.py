"""Rolling-origin backtest: the only place accuracy and calibration are measured.

For every test origin T the data is physically truncated at T, the full model is
refitted on what was observable then, and the 30/60/90-day totals that followed
are forecast and compared with what actually happened.  Nothing after T - not
targets, not seasonal analogs, not calibration residuals - can reach the model.

The out-of-sample residuals collected here are also what the final model's
conformal calibration is built from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BACKTEST_MIN_TRAIN_DAYS,
    BACKTEST_STEP_DAYS,
    HORIZONS,
    INTERVAL_COVERAGE,
    SEASONAL_CLIP,
    TRAIN_ORIGIN_STEP_DAYS,
)
from src.features import FeatureBuilder, _cumsum, _wsum, build_samples
from src.forecast import META_COLS, apply_spend_plan, assemble
from src.hierarchy import Hierarchy
from src.model import apply_conformal, fit_conformal, fit_core, raw_scores

LEVELS = ["blended", "channel", "campaign_type", "campaign"]
MIN_PRIOR_ORIGINS = 8          # completed earlier origins needed before coverage is scored
ROAS_EVAL_MIN_SPEND = 50.0     # realised horizon spend below this is excluded from ROAS metrics


def training_origins(last_idx: int, step: int = TRAIN_ORIGIN_STEP_DAYS) -> list[int]:
    """Forecast origins used for fitting, walking back from the newest usable day."""
    return list(range(last_idx - step, 40, -step))


def test_origins(hier: Hierarchy, step: int = BACKTEST_STEP_DAYS) -> list[int]:
    """Weekly Monday origins with enough history behind and 30 days of truth ahead."""
    last = hier.n_days - 1 - min(HORIZONS)
    idx = [i for i in range(BACKTEST_MIN_TRAIN_DAYS, last + 1) if hier.dates[i].weekday() == 0]
    return idx[::max(step // 7, 1)]


def run_backtest(hier: Hierarchy, origins: list[int] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Refit at every test origin on truncated data; return out-of-sample scored rows."""
    origins = test_origins(hier) if origins is None else origins
    cs_rev, cs_sp = _cumsum(hier.revenue), _cumsum(hier.spend)
    meta = hier.meta[META_COLS]
    rows = []
    for n, t in enumerate(origins, 1):
        past = hier.truncate(hier.dates[t])
        core = fit_core(build_samples(past, training_origins(t)))
        test = FeatureBuilder(past).build(t, with_targets=False).drop(columns=["y_rev", "y_sp"])
        actual = pd.concat([
            pd.DataFrame({"series_id": hier.meta["series_id"], "horizon": h,
                          "y_rev": _wsum(cs_rev, t + 1, t + h), "y_sp": _wsum(cs_sp, t + 1, t + h)})
            for h in HORIZONS if t + h <= hier.n_days - 1
        ])
        test = test.merge(actual, on=["series_id", "horizon"], how="inner").merge(meta, on="series_id")
        rows.append(raw_scores(core, test))
        if verbose:
            print(f"[backtest] {n:>2}/{len(origins)}  origin {hier.dates[t].date()}  "
                  f"train rows {core['n_samples']:,}", flush=True)
    return pd.concat(rows, ignore_index=True)


def honest_forecasts(oos: pd.DataFrame) -> pd.DataFrame:
    """Forecast tables as they would have been issued at each origin.

    Intervals at origin T are calibrated only with residuals from earlier origins
    whose target windows had fully elapsed by T.  The budget-informed forecast
    re-runs each origin with the spend that was actually deployed.
    """
    done = oos["origin"] + pd.to_timedelta(oos["horizon"], unit="D")
    frames = []
    for origin, cur in oos.groupby("origin"):
        prior = oos[done <= origin]
        conformal = fit_conformal(prior) if len(prior) else {}
        base = assemble(apply_conformal(cur, conformal))
        key = cur.set_index(["series_id", "horizon"])
        idx = pd.MultiIndex.from_arrays([base["series_id"], base["horizon_days"]])
        base["actual_revenue"] = key["y_rev"].reindex(idx).to_numpy()
        base["actual_spend"] = key["y_sp"].reindex(idx).to_numpy()
        base["peak"] = key["peaky"].reindex(idx).to_numpy()
        base["seasonal_naive"] = (key["B_rev"] * np.exp(key["seas_port_rev"].fillna(0).clip(*SEASONAL_CLIP))
                                  ).reindex(idx).to_numpy()
        base["n_prior_origins"] = base["horizon_days"].map(
            prior.groupby("horizon")["origin"].nunique()).fillna(0).to_numpy()

        types = base[base["level"] == "campaign_type"]
        plan = pd.Series(types["actual_spend"].to_numpy(),
                         index=pd.MultiIndex.from_arrays([types["series_id"], types["horizon_days"]]))
        scen = apply_spend_plan(base, plan)
        for q in ("p10", "p50", "p90"):
            base[f"budget_revenue_{q}"] = scen[f"revenue_{q}"].to_numpy()
        frames.append(base)
    return pd.concat(frames, ignore_index=True)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.abs(actual).sum())
    return float(np.abs(actual - pred).sum() / denom * 100) if denom > 0 else float("nan")


def pinball(actual: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = actual - pred
    return float(np.maximum(q * diff, (q - 1) * diff).sum())


def _coverage(actual, lo, hi) -> float:
    return float(((actual >= lo) & (actual <= hi)).mean() * 100) if len(actual) else float("nan")


def _metrics(g: pd.DataFrame) -> dict:
    a = g["actual_revenue"].to_numpy()
    lo_q, hi_q = (1 - INTERVAL_COVERAGE) / 2, (1 + INTERVAL_COVERAGE) / 2
    total = max(float(a.sum()), 1e-9)
    cal = g[g["n_prior_origins"] >= MIN_PRIOR_ORIGINS]
    roas = g[g["actual_spend"] >= ROAS_EVAL_MIN_SPEND]
    roas_cal = roas[roas["n_prior_origins"] >= MIN_PRIOR_ORIGINS]
    return {
        "n": int(len(g)),
        "wape": wape(a, g["revenue_p50"].to_numpy()),
        "wape_budget_known": wape(a, g["budget_revenue_p50"].to_numpy()),
        "wape_run_rate": wape(a, g["baseline_revenue"].to_numpy()),
        "wape_seasonal_naive": wape(a, g["seasonal_naive"].to_numpy()),
        "bias_pct": float((g["revenue_p50"].sum() / total - 1) * 100),
        "coverage": _coverage(cal["actual_revenue"], cal["revenue_p10"], cal["revenue_p90"]),
        "coverage_budget_known": _coverage(cal["actual_revenue"], cal["budget_revenue_p10"],
                                           cal["budget_revenue_p90"]),
        "n_coverage": int(len(cal)),
        "interval_width_pct": float((g["revenue_p90"] - g["revenue_p10"]).sum() / total * 100),
        "pinball_pct": float((pinball(a, g["revenue_p10"].to_numpy(), lo_q)
                              + pinball(a, g["revenue_p50"].to_numpy(), 0.5)
                              + pinball(a, g["revenue_p90"].to_numpy(), hi_q)) / 3 / total * 100),
        "spend_wape": wape(g["actual_spend"].to_numpy(), g["spend_p50"].to_numpy()),
        # revenue error if the realised spend had been known and only ROAS was forecast
        "roas_wape": wape(roas["actual_revenue"].to_numpy(),
                          (roas["roas_p50"] * roas["actual_spend"]).to_numpy()),
        "roas_coverage": _coverage(roas_cal["actual_revenue"] / roas_cal["actual_spend"],
                                   roas_cal["roas_p10"], roas_cal["roas_p90"]),
    }


def summarise(fc: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Accuracy and calibration by level and horizon (optionally split further)."""
    by = by or []
    keys = by + ["level", "horizon_days"]
    out = pd.DataFrame([{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **_metrics(g)}
                        for k, g in fc.groupby(keys)])
    out["_o"] = out["level"].map({l: i for i, l in enumerate(LEVELS)})
    return out.sort_values(by + ["_o", "horizon_days"]).drop(columns="_o").reset_index(drop=True)


def compare_with_legacy(fc: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    """v4 vs the v3.1 pipeline on the origins and series both were run on.

    `legacy` holds the v3.1 pipeline's own rolling-origin forecasts (retrained at
    every origin exactly as its train.py did) with realised revenue attached.
    Its native 28/56/91-day windows are rescaled to 30/60/90 days.
    """
    leg = legacy.copy()
    leg["origin"] = pd.to_datetime(leg["origin"])
    scale = leg["h"].map({30: 30 / 28, 60: 60 / 56, 90: 90 / 91})
    for c in ("p10", "p50", "p90"):
        leg[c] = leg[c] * scale
    leg = leg.rename(columns={"h": "horizon_days"})
    leg["campaign_name"] = leg["campaign_name"].fillna("")
    leg["campaign_type"] = leg["campaign_type"].fillna("")
    keys = ["origin", "level", "channel", "campaign_type", "campaign_name", "horizon_days"]
    both = fc.merge(leg[keys + ["p10", "p50", "p90"]], on=keys, how="inner")
    rows = []
    for (level, h), g in both.groupby(["level", "horizon_days"]):
        a = g["actual_revenue"].to_numpy()
        rows.append({
            "level": level, "horizon_days": int(h), "n": int(len(g)), "origins": int(g["origin"].nunique()),
            "wape_v31": wape(a, g["p50"].to_numpy()), "wape_v4": wape(a, g["revenue_p50"].to_numpy()),
            "coverage_v31": _coverage(a, g["p10"], g["p90"]),
            "coverage_v4": _coverage(a, g["revenue_p10"], g["revenue_p90"]),
            "bias_pct_v31": float((g["p50"].sum() / max(a.sum(), 1e-9) - 1) * 100),
            "bias_pct_v4": float((g["revenue_p50"].sum() / max(a.sum(), 1e-9) - 1) * 100),
        })
    out = pd.DataFrame(rows)
    out["_o"] = out["level"].map({l: i for i, l in enumerate(LEVELS)})
    return out.sort_values(["_o", "horizon_days"]).drop(columns="_o").reset_index(drop=True)
