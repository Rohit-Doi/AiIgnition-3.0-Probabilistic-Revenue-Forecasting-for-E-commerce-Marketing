"""Turn feature rows + a model bundle into coherent P10/P50/P90 forecasts.

Two modes share one code path:

* baseline  - no budget is supplied; spend is forecast alongside revenue and the
              intervals carry both demand and budget uncertainty.
* scenario  - a spend plan is supplied; revenue moves with the plan through the
              fitted elasticity and the intervals narrow to demand uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ROAS_CAP, ROAS_MIN_SPEND
from src.features import SeasonalReference, fill_seasonal_from_reference
from src.model import DRIVER_GROUPS, apply_conformal, raw_scores

META_COLS = ["series_id", "level", "channel", "campaign_type", "campaign_name", "parent_id"]
DRIVER_COLS = [f"drv_{g}" for g in DRIVER_GROUPS] + ["drv_reconcile", "drv_budget"]

SUBMISSION_COLUMNS = [
    "channel", "campaign_type", "campaign_name", "horizon_days",
    "p10_revenue", "p50_revenue", "p90_revenue", "p10_roas", "p50_roas", "p90_roas",
]
_LEVEL_ORDER = {"blended": 0, "channel": 1, "campaign_type": 2, "campaign": 3}


def _from_log(base: np.ndarray, c: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Invert the smoothed log-ratio transform back to dollars."""
    return np.maximum((base + c) * np.exp(z) - c, 0.0)


def _top_down(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """Proportional top-down reconciliation within one horizon.

    Channels are scaled to add up to the blended forecast and campaign types to
    their channel.  Campaigns are only ever scaled *down*: a campaign type's
    future revenue includes campaigns that do not exist yet, so existing
    campaigns may legitimately sum to less than their type.
    """
    v = pd.Series(values, index=frame["series_id"].to_numpy(), dtype=float)
    parent = frame["parent_id"].to_numpy()
    sid = frame["series_id"].to_numpy()
    for level, shrink_only in (("channel", False), ("campaign_type", False), ("campaign", True)):
        rows = (frame["level"] == level).to_numpy()
        if not rows.any():
            continue
        child = v.loc[sid[rows]]
        siblings = child.groupby(parent[rows]).transform("sum").to_numpy()
        par = v.reindex(parent[rows]).to_numpy()
        ok = (siblings > 0) & np.isfinite(par)
        factor = np.where(ok, par / np.where(siblings > 0, siblings, 1.0), 1.0)
        if shrink_only:
            factor = np.minimum(factor, 1.0)
        v.loc[sid[rows]] = child.to_numpy() * factor
    return v.loc[sid].to_numpy()


def _non_decreasing_in_horizon(series_id: np.ndarray, horizon: np.ndarray, values: np.ndarray) -> np.ndarray:
    """A longer window cannot total less than a shorter one: running maximum across horizons."""
    frame = pd.DataFrame({"sid": series_id, "h": horizon, "v": values})
    ordered = frame.sort_values(["sid", "h"], kind="stable")
    return ordered.groupby("sid")["v"].cummax().reindex(frame.index).to_numpy()


def _reconcile(d: pd.DataFrame, col: str) -> np.ndarray:
    out = np.zeros(len(d))
    for h in d["horizon"].unique():
        m = (d["horizon"] == h).to_numpy()
        out[m] = _top_down(d[m], d.loc[m, col].to_numpy())
    return out


def score(bundle: dict, feats: pd.DataFrame) -> pd.DataFrame:
    """Stage 1–3 outputs in log space for every feature row."""
    reference = bundle.get("seasonal_reference")
    reference = SeasonalReference.from_dict(reference) if reference else None
    feats = fill_seasonal_from_reference(feats, feats["channel"], reference)
    return apply_conformal(raw_scores(bundle, feats), bundle["conformal"])


def assemble(scored: pd.DataFrame) -> pd.DataFrame:
    """Baseline forecast table (no budget supplied) from scored feature rows."""
    d = scored.reset_index(drop=True).copy()
    alive = d["alive"].to_numpy()
    B_rev, C_rev = d["B_rev"].to_numpy(), d["C_rev"].to_numpy()
    B_sp, C_sp = d["B_sp"].to_numpy(), d["C_sp"].to_numpy()

    sid, hz = d["series_id"].to_numpy(), d["horizon"].to_numpy()
    raw_rev = np.where(alive, _from_log(B_rev, C_rev, d["z1_rev"].to_numpy()), 0.0)
    raw_sp = np.where(alive, _from_log(B_sp, C_sp, d["z1_sp"].to_numpy()), 0.0)
    d["rev_raw"] = _non_decreasing_in_horizon(sid, hz, raw_rev)
    d["sp_raw"] = _non_decreasing_in_horizon(sid, hz, raw_sp)
    rev50 = _reconcile(d, "rev_raw")
    sp50 = _reconcile(d, "sp_raw")
    d["drv_reconcile"] = np.where(alive, np.log((rev50 + C_rev) / (raw_rev + C_rev)), 0.0)
    d["drv_budget"] = 0.0

    def bounds(mid: np.ndarray, c: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p10 = _non_decreasing_in_horizon(sid, hz, np.where(alive, _from_log(mid, c, lo), 0.0))
        p90 = _non_decreasing_in_horizon(sid, hz, np.where(alive, _from_log(mid, c, hi), 0.0))
        return np.minimum(p10, mid), np.maximum(p90, mid)

    out = d[META_COLS].copy()
    out["horizon_days"] = d["horizon"].astype(int)
    out["origin"] = d["origin"]
    out["revenue_p10"], out["revenue_p90"] = bounds(rev50, C_rev, d["lo_rev"].to_numpy(), d["hi_rev"].to_numpy())
    out["revenue_p50"] = rev50
    out["spend_p10"], out["spend_p90"] = bounds(sp50, C_sp, d["lo_sp"].to_numpy(), d["hi_sp"].to_numpy())
    out["spend_p50"] = sp50

    # ROAS bounds come from the ROAS residual model (revenue and spend errors are
    # correlated, so this band is tighter than revenue band / spend), inverted
    # through the same smoothed transform with spend held at its median.
    has_spend = sp50 >= ROAS_MIN_SPEND
    safe_sp = np.where(has_spend, sp50, 1.0)
    roas = {"p50": rev50 / safe_sp,
            "p10": np.minimum(_from_log(rev50, C_rev, d["lo_roas"].to_numpy()), rev50) / safe_sp,
            "p90": np.maximum(_from_log(rev50, C_rev, d["hi_roas"].to_numpy()), rev50) / safe_sp}
    for q, v in roas.items():
        out[f"roas_{q}"] = np.clip(np.where(has_spend & alive, v, 0.0), 0, ROAS_CAP)

    out["baseline_revenue"] = np.where(alive, B_rev, 0.0)      # 28-day run-rate x horizon
    out["baseline_spend"] = np.where(alive, B_sp, 0.0)
    out["elasticity"] = d["beta"].to_numpy()
    out["alive"] = alive
    out["mode"] = "baseline"
    for c in DRIVER_COLS:
        out[c] = np.where(alive, d[c].to_numpy(), 0.0)
    # Kept for scenario maths and explanations.  Once spend is fixed by a plan,
    # revenue = spend x ROAS, so the revenue band under a plan is the ROAS band.
    out["_c_rev"], out["_c_sp"] = C_rev, C_sp
    out["_lo_budget"], out["_hi_budget"] = d["lo_roas"].to_numpy(), d["hi_roas"].to_numpy()
    return _sorted(out)


def _sorted(fc: pd.DataFrame) -> pd.DataFrame:
    order = fc["level"].map(_LEVEL_ORDER)
    return (fc.assign(_o=order)
            .sort_values(["_o", "channel", "campaign_type", "campaign_name", "horizon_days"], kind="stable")
            .drop(columns="_o").reset_index(drop=True))


def forecast(bundle: dict, feats: pd.DataFrame) -> pd.DataFrame:
    """Baseline P10/P50/P90 revenue, spend and ROAS for every series and horizon."""
    return assemble(score(bundle, feats))


# ─── Budget scenarios ────────────────────────────────────────────────────────

def apply_spend_plan(baseline: pd.DataFrame, plan: pd.Series) -> pd.DataFrame:
    """Re-forecast under a spend plan.

    `plan` maps (series_id, horizon_days) of *campaign-type* rows to planned
    spend over that horizon.  Types without an entry keep their expected spend.
    Revenue responds at the campaign-type level through the fitted elasticity,

        revenue(S) = (revenue_0 + c) * ((S + c_s) / (S_0 + c_s)) ** elasticity - c,

    channels and the blended total are rebuilt bottom-up, and campaigns move in
    proportion to their type.  With the plan equal to expected spend the result
    is identical to the baseline.
    """
    fc = baseline.copy()
    key = pd.MultiIndex.from_arrays([fc["series_id"], fc["horizon_days"]])
    is_type = (fc["level"] == "campaign_type").to_numpy()

    sp0 = fc["spend_p50"].to_numpy()
    rev0 = fc["revenue_p50"].to_numpy()
    planned = np.asarray(plan.reindex(key).to_numpy(), dtype=float)
    sp_new = np.where(is_type & np.isfinite(planned), np.maximum(planned, 0.0), sp0)
    c_rev, c_sp = fc["_c_rev"].to_numpy(), fc["_c_sp"].to_numpy()

    delta = np.where(is_type, np.log((sp_new + c_sp) / (sp0 + c_sp)), 0.0)
    rev_new = np.where(is_type, _from_log(rev0, c_rev, fc["elasticity"].to_numpy() * delta), rev0)
    rev_new = np.where(is_type & (sp_new <= 0), 0.0, rev_new)

    fc["_rev"], fc["_sp"] = rev_new, sp_new
    types = fc[is_type]
    for h, grp in fc.groupby("horizon_days"):
        t = types[types["horizon_days"] == h]
        by_channel = t.groupby("parent_id")[["_rev", "_sp"]].sum()
        ch = grp.index[grp["level"] == "channel"]
        fc.loc[ch, ["_rev", "_sp"]] = by_channel.reindex(fc.loc[ch, "series_id"]).to_numpy()
        bl = grp.index[grp["level"] == "blended"]
        fc.loc[bl, ["_rev", "_sp"]] = by_channel.sum().to_numpy()
        # campaigns follow their type
        ratio = t.set_index("series_id")
        r_rev = (ratio["_rev"] / ratio["revenue_p50"].replace(0, np.nan)).fillna(0.0)
        r_sp = (ratio["_sp"] / ratio["spend_p50"].replace(0, np.nan)).fillna(0.0)
        cp = grp.index[grp["level"] == "campaign"]
        fc.loc[cp, "_rev"] = fc.loc[cp, "revenue_p50"].to_numpy() * r_rev.reindex(fc.loc[cp, "parent_id"]).to_numpy()
        fc.loc[cp, "_sp"] = fc.loc[cp, "spend_p50"].to_numpy() * r_sp.reindex(fc.loc[cp, "parent_id"]).to_numpy()

    rev, sp = fc["_rev"].fillna(0.0).to_numpy(), fc["_sp"].fillna(0.0).to_numpy()
    alive = fc["alive"].to_numpy() | (sp > 0)
    fc["drv_budget"] = np.where(alive, np.log((rev + c_rev) / (rev0 + c_rev)), 0.0)
    fc["revenue_p50"] = rev
    fc["revenue_p10"] = np.where(alive, np.minimum(_from_log(rev, c_rev, fc["_lo_budget"].to_numpy()), rev), 0.0)
    fc["revenue_p90"] = np.where(alive, np.maximum(_from_log(rev, c_rev, fc["_hi_budget"].to_numpy()), rev), 0.0)
    for q in ("p10", "p50", "p90"):
        fc[f"spend_{q}"] = sp
        ok = sp >= ROAS_MIN_SPEND
        fc[f"roas_{q}"] = np.clip(np.where(ok, fc[f"revenue_{q}"] / np.where(ok, sp, 1.0), 0.0), 0, ROAS_CAP)
    fc["mode"] = "scenario"
    return fc.drop(columns=["_rev", "_sp"])


# ─── Output contract ─────────────────────────────────────────────────────────

def to_submission(fc: pd.DataFrame) -> pd.DataFrame:
    """The scored file: one row per (channel, campaign type, campaign, horizon).

    Blended rows carry channel 'all'; channel and blended rows leave
    campaign_type / campaign_name empty.
    """
    out = pd.DataFrame({
        "channel": fc["channel"],
        "campaign_type": fc["campaign_type"],
        "campaign_name": fc["campaign_name"],
        "horizon_days": fc["horizon_days"].astype(int),
        "p10_revenue": fc["revenue_p10"].round(2),
        "p50_revenue": fc["revenue_p50"].round(2),
        "p90_revenue": fc["revenue_p90"].round(2),
        "p10_roas": fc["roas_p10"].round(4),
        "p50_roas": fc["roas_p50"].round(4),
        "p90_roas": fc["roas_p90"].round(4),
    })
    return out[SUBMISSION_COLUMNS]


def public_columns(fc: pd.DataFrame) -> pd.DataFrame:
    """Forecast table without internal helper columns."""
    return fc[[c for c in fc.columns if not c.startswith("_")]]

