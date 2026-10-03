"""Why the forecast says what it says: drivers, anomalies and risks as plain facts.

Nothing here calls an LLM.  The functions turn the model's own arithmetic into a
compact, numeric fact sheet; src/insights.py hands that sheet to an LLM (or to a
deterministic writer) so every sentence it produces can be traced to a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MATURITY_LAG_DAYS, PEAK_SEASON
from src.hierarchy import Hierarchy
from src.scenario import marginal_roas

MATERIAL_SPEND_SHARE = 0.03    # a series must carry this share of portfolio spend to be screened

DRIVER_LABELS = {
    "drv_momentum": "Recent momentum beyond normal noise (last 7-14 days vs 28)",
    "drv_reversion": "Pull towards the longer-run level (56-182 days)",
    "drv_seasonality": "Seasonality (same window in other years)",
    "drv_efficiency": "ROAS drift (28 vs 91 days)",
    "drv_context": "Portfolio-wide momentum (last 7 days vs 28)",
    "drv_structural": "Inactive-day adjustment",
    "drv_reconcile": "Hierarchy reconciliation",
    "drv_budget": "Budget plan vs expected spend",
}


# Compact labels for charts (same order as DRIVER_LABELS).
DRIVER_SHORT = {
    "drv_momentum": "Momentum (7-14d vs 28d)",
    "drv_reversion": "Longer-run level",
    "drv_seasonality": "Seasonality",
    "drv_efficiency": "ROAS drift",
    "drv_context": "Portfolio momentum",
    "drv_structural": "Inactive days",
    "drv_reconcile": "Reconciliation",
    "drv_budget": "Budget plan",
}


# ─── Driver waterfall ────────────────────────────────────────────────────────

def driver_waterfall(row: pd.Series) -> pd.DataFrame:
    """Exact multiplicative decomposition of one forecast, expressed in dollars.

    The stage-1 model is linear in logs, so the P50 equals the run-rate baseline
    times exp(sum of driver contributions).  Steps are applied in a fixed order
    and each step's dollar effect is the change it causes at that point.
    """
    c = float(row.get("_c_rev", 0.0))
    level = float(row["baseline_revenue"])
    steps = [{"driver": "Run-rate baseline (last 28 days x horizon)", "short": "Run-rate baseline",
              "effect": level, "kind": "start", "pct": 0.0}]
    for col, label in DRIVER_LABELS.items():
        contribution = float(row.get(col, 0.0) or 0.0)
        new_level = max((level + c) * float(np.exp(contribution)) - c, 0.0)
        if abs(new_level - level) > 1e-9:
            steps.append({"driver": label, "short": DRIVER_SHORT[col], "effect": new_level - level,
                          "kind": "step", "pct": (np.exp(contribution) - 1) * 100})
        level = new_level
    steps.append({"driver": "Forecast P50", "short": "Forecast P50", "effect": float(row["revenue_p50"]),
                  "kind": "total", "pct": 0.0})
    return pd.DataFrame(steps)


# ─── Anomalies in recent history ─────────────────────────────────────────────

def weekly_history(hier: Hierarchy, series_id: int, weeks: int = 104) -> pd.DataFrame:
    """Complete trailing 7-day blocks ending at the forecast origin."""
    n = min(weeks, hier.n_days // 7)
    end = hier.n_days
    rows = []
    for k in range(n, 0, -1):
        a, b = end - 7 * k, end - 7 * (k - 1)
        rows.append({"week_end": hier.dates[b - 1], "revenue": float(hier.revenue[series_id, a:b].sum()),
                     "spend": float(hier.spend[series_id, a:b].sum())})
    out = pd.DataFrame(rows)
    out["roas"] = np.where(out["spend"] > 0, out["revenue"] / out["spend"].replace(0, np.nan), np.nan)
    return out


def yearly_overlay(hier: Hierarchy, series_id: int) -> pd.DataFrame:
    """Weekly revenue and spend by ISO week for each calendar year (seasonality at a glance)."""
    daily = pd.DataFrame({"revenue": hier.revenue[series_id], "spend": hier.spend[series_id]}, index=hier.dates)
    weekly = daily.resample("W-SUN").sum()
    weekly = weekly.iloc[:-1] if len(weekly) > 1 else weekly       # last week is incomplete
    iso = weekly.index.isocalendar()
    return pd.DataFrame({"year": iso["year"].to_numpy(), "week": iso["week"].to_numpy(),
                         "week_end": weekly.index, "revenue": weekly["revenue"].to_numpy(),
                         "spend": weekly["spend"].to_numpy()})


def movers(fc: pd.DataFrame, horizon: int, level: str = "campaign_type", n: int = 5) -> pd.DataFrame:
    """Series whose forecast departs most, in dollars, from their own run-rate."""
    h = fc[(fc["horizon_days"] == horizon) & (fc["level"] == level) & fc["alive"]].copy()
    h["change_usd"] = h["revenue_p50"] - h["baseline_revenue"]
    h["change_pct"] = (h["revenue_p50"] / h["baseline_revenue"].replace(0, np.nan) - 1) * 100
    h["_abs"] = h["change_usd"].abs()
    return h.sort_values("_abs", ascending=False).head(n).drop(columns="_abs").reset_index(drop=True)


def detect_anomalies(hier: Hierarchy, lookback_weeks: int = 26, window: int = 8,
                     threshold: float = 3.0) -> pd.DataFrame:
    """Weeks where revenue, spend or ROAS broke away from their own recent behaviour.

    For each channel and campaign type, a week is compared with the median of the
    preceding `window` weeks on a log scale, in units of that window's robust
    spread (MAD).  The final, still-maturing week is skipped.
    """
    columns = ["week_ending", "level", "channel", "campaign_type", "metric", "direction", "value",
               "typical", "robust_z", "impact_usd", "spend", "revenue"]
    records = []
    n_weeks = lookback_weeks + window + 1
    portfolio_spend = max(float(weekly_history(hier, 0, n_weeks)["spend"].tail(lookback_weeks).sum()), 1e-9)
    meta = hier.meta[hier.meta["level"].isin(["channel", "campaign_type"])]
    for _, m in meta.iterrows():
        wk = weekly_history(hier, int(m["series_id"]), n_weeks)
        if len(wk) < window + 2:
            continue
        # only series that matter to the portfolio: tiny ones are all noise
        if wk["spend"].tail(lookback_weeks).sum() < MATERIAL_SPEND_SHARE * portfolio_spend:
            continue
        wk = wk.iloc[:-1] if MATURITY_LAG_DAYS > 0 else wk
        typical_roas = wk["revenue"].sum() / max(wk["spend"].sum(), 1e-9)
        for metric in ("revenue", "spend", "roas"):
            floor = 0.05 if metric == "roas" else max(0.05 * wk[metric].median(), 1.0)
            x = np.log(wk[metric].fillna(0).clip(lower=floor))
            med = x.rolling(window, min_periods=window).median().shift(1)
            mad = (x - med).abs().rolling(window, min_periods=window).median().shift(1)
            z = (x - med) / (1.4826 * mad.clip(lower=0.15))
            for i in z.index[-lookback_weeks:]:
                if not np.isfinite(z[i]) or abs(z[i]) < threshold:
                    continue
                value = float(wk.loc[i, metric]) if np.isfinite(wk.loc[i, metric]) else 0.0
                typical = float(np.exp(med[i]))
                if metric == "roas":
                    if wk.loc[i, "spend"] < 0.25 * wk["spend"].median():
                        continue      # ROAS on a near-zero spend week is noise
                    impact = (value - typical) * float(wk.loc[i, "spend"])
                elif metric == "spend":
                    impact = (value - typical) * typical_roas
                else:
                    impact = value - typical
                records.append({
                    "week_ending": str(wk.loc[i, "week_end"].date()),
                    "level": m["level"], "channel": m["channel"], "campaign_type": m["campaign_type"],
                    "metric": metric, "direction": "above" if z[i] > 0 else "below",
                    "value": value, "typical": typical, "robust_z": float(z[i]), "impact_usd": float(impact),
                    "spend": float(wk.loc[i, "spend"]), "revenue": float(wk.loc[i, "revenue"]),
                })
    if not records:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(records)
    # one row per series-week: the metric that broke away the most
    out["_abs"] = out["robust_z"].abs()
    out = (out.sort_values("_abs", ascending=False)
           .drop_duplicates(["week_ending", "channel", "campaign_type"]))
    out["_impact"] = out["impact_usd"].abs()
    return out.sort_values("_impact", ascending=False)[columns].reset_index(drop=True)


# ─── Risk flags ──────────────────────────────────────────────────────────────

def _touches_peak(origin: pd.Timestamp, horizon: int) -> bool:
    (m0, d0), (m1, d1) = PEAK_SEASON
    days = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    md = days.month * 100 + days.day
    return bool(((md >= m0 * 100 + d0) & (md <= m1 * 100 + d1)).any())


def risk_flags(fc: pd.DataFrame, hier: Hierarchy, report: dict, horizon: int,
               breakeven_roas: float = 1.0) -> list[dict]:
    """Operational risks an analyst should see before trusting or acting on the forecast."""
    flags: list[dict] = []

    def add(severity: str, title: str, detail: str) -> None:
        flags.append({"severity": severity, "title": title, "detail": detail})

    h = fc[fc["horizon_days"] == horizon]
    blended = h[h["level"] == "blended"].iloc[0]
    channels = h[h["level"] == "channel"]
    types = h[(h["level"] == "campaign_type") & (h["spend_p50"] > 0)]
    campaigns = h[h["level"] == "campaign"]
    total = max(float(blended["revenue_p50"]), 1e-9)

    spread = (blended["revenue_p90"] - blended["revenue_p10"]) / total
    if spread > 1.2:
        add("high", "Wide forecast range",
            f"The {horizon}-day P10-P90 range spans {spread * 100:.0f}% of the median "
            f"(${blended['revenue_p10']:,.0f} to ${blended['revenue_p90']:,.0f}). Plan against the range, not the midpoint.")
    downside = 1 - blended["revenue_p10"] / total
    if downside > 0.5:
        add("high", "Deep downside scenario",
            f"The P10 outcome is {downside * 100:.0f}% below the median. Past budget pull-backs after promotions "
            "produced drops of this size.")

    momentum = float(blended.get("drv_momentum", 0.0))
    if momentum < -0.15:
        add("high", "Revenue is decelerating",
            f"The last 7-14 days run {abs(np.expm1(momentum)) * 100:.0f}% below the 28-day pace, "
            "which pulls the forecast under the run-rate.")
    elif momentum > 0.15:
        add("medium", "Recent surge may not persist",
            f"The last 7-14 days run {np.expm1(momentum) * 100:.0f}% above the 28-day pace; the forecast "
            "assumes only part of that surge carries forward.")

    if _touches_peak(hier.origin, horizon):
        add("high", "Window includes peak season",
            "Peak-season revenue is driven by budget decisions the history cannot reveal. Use the budget "
            "simulator with the planned peak budget; the baseline relies on prior-year analogs.")

    weak = types[(types["roas_p50"] < breakeven_roas) & (types["spend_p50"] > 0.03 * blended["spend_p50"])]
    for _, r in weak.iterrows():
        add("medium", f"{r['channel']} {r['campaign_type']} below break-even",
            f"Expected ROAS {r['roas_p50']:.2f} on ${r['spend_p50']:,.0f} of spend "
            f"(P10-P90 {r['roas_p10']:.2f}-{r['roas_p90']:.2f}).")
    at_risk = types[(types["roas_p10"] < breakeven_roas) & (types["roas_p50"] >= breakeven_roas)
                    & (types["spend_p50"] > 0.05 * blended["spend_p50"])]
    for _, r in at_risk.iterrows():
        add("low", f"{r['channel']} {r['campaign_type']} could fall below break-even",
            f"Median ROAS {r['roas_p50']:.2f} but the P10 case is {r['roas_p10']:.2f}.")

    if len(channels):
        top = channels.sort_values("revenue_p50", ascending=False).iloc[0]
        share = top["revenue_p50"] / total
        if share > 0.75:
            add("medium", "Channel concentration",
                f"{top['channel'].title()} carries {share * 100:.0f}% of forecast revenue.")
    if len(campaigns):
        top = campaigns.sort_values("revenue_p50", ascending=False).iloc[0]
        share = top["revenue_p50"] / total
        if share > 0.20:
            add("medium", "Campaign concentration",
                f"{top['campaign_name']} ({top['channel']}) carries {share * 100:.0f}% of forecast revenue.")

    age_days = (hier.spend > 0).argmax(axis=1)
    new = hier.meta[(hier.meta["level"] == "campaign").to_numpy()
                    & (hier.n_days - age_days <= 28) & (hier.spend.sum(axis=1) > 0)]
    if len(new):
        add("low", f"{len(new)} campaigns launched in the last 28 days",
            "New campaigns have little history; their individual forecasts are directional.")

    if hier.n_days < 392:
        add("medium", "Short history",
            f"Only {hier.n_days} days of data: seasonality comes from the model's stored reference profile, "
            "not from this account's own prior year.")
    coverage = report.get("channel_coverage", [])
    if any(c.get("days_behind_newest", 0) > 0 for c in coverage):
        add("low", "Channel exports end on different days",
            ", ".join(f"{c['channel']} {c['last_date']}" for c in coverage)
            + f". The forecast starts after {hier.origin.date()}, the last day every channel covers.")
    if report.get("spend_without_revenue_campaigns", 0):
        add("low", "Spend with no attributed revenue",
            f"{report['spend_without_revenue_campaigns']} campaigns spent $100+ with zero tracked revenue "
            "(upper-funnel campaigns or a tracking gap).")

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(flags, key=lambda f: order[f["severity"]])


# ─── Fact sheet for the insight layer ────────────────────────────────────────

def _r(x: float, nd: int = 0) -> float:
    return float(np.round(float(x), nd))


def build_facts(fc: pd.DataFrame, hier: Hierarchy, report: dict, bundle: dict, horizon: int,
                scenario: pd.DataFrame | None = None, allocation: pd.DataFrame | None = None,
                anomalies: pd.DataFrame | None = None, risks: list[dict] | None = None) -> dict:
    """Everything the insight layer is allowed to talk about, as numbers."""
    h = fc[fc["horizon_days"] == horizon]
    blended = h[h["level"] == "blended"].iloc[0]
    blended_wk = weekly_history(hier, 0, 8)

    def row_facts(r: pd.Series) -> dict:
        return {
            "revenue_p10": _r(r["revenue_p10"]), "revenue_p50": _r(r["revenue_p50"]),
            "revenue_p90": _r(r["revenue_p90"]), "spend_p50": _r(r["spend_p50"]),
            "roas_p10": _r(r["roas_p10"], 2), "roas_p50": _r(r["roas_p50"], 2), "roas_p90": _r(r["roas_p90"], 2),
            "run_rate_revenue": _r(r["baseline_revenue"]),
            "vs_run_rate_pct": _r((r["revenue_p50"] / max(r["baseline_revenue"], 1e-9) - 1) * 100, 1),
            "spend_elasticity": _r(r["elasticity"], 2),
        }

    wf = driver_waterfall(blended)
    drivers = [{"driver": s["driver"], "effect_usd": _r(s["effect"]), "effect_pct": _r(s["pct"], 1)}
               for _, s in wf.iterrows() if s["kind"] == "step"]

    facts: dict = {
        "forecast_origin": str(hier.origin.date()),
        "horizon_days": int(horizon),
        "window": f"{(hier.origin + pd.Timedelta(days=1)).date()} to {(hier.origin + pd.Timedelta(days=horizon)).date()}",
        "history": {"first_date": str(hier.dates[0].date()), "days": int(hier.n_days),
                    "last_8_weeks_revenue": [_r(v) for v in blended_wk["revenue"]],
                    "last_8_weeks_spend": [_r(v) for v in blended_wk["spend"]]},
        "blended": row_facts(blended),
        "blended_all_horizons": [
            {"horizon_days": int(r["horizon_days"]), **row_facts(r)}
            for _, r in fc[fc["level"] == "blended"].iterrows()],
        "drivers_of_blended_forecast": drivers,
        "channels": [{"channel": r["channel"], **row_facts(r),
                      "share_of_revenue_pct": _r(r["revenue_p50"] / max(blended["revenue_p50"], 1e-9) * 100, 1)}
                     for _, r in h[h["level"] == "channel"].iterrows()],
        "campaign_types": [{"channel": r["channel"], "campaign_type": r["campaign_type"], **row_facts(r)}
                           for _, r in h[(h["level"] == "campaign_type") & (h["spend_p50"] > 0)]
                           .sort_values("revenue_p50", ascending=False).iterrows()],
        "top_campaigns": [{"channel": r["channel"], "campaign": r["campaign_name"],
                           "revenue_p50": _r(r["revenue_p50"]), "roas_p50": _r(r["roas_p50"], 2)}
                          for _, r in h[h["level"] == "campaign"].nlargest(5, "revenue_p50").iterrows()],
        "marginal_roas": [{"channel": r["channel"], "campaign_type": r["campaign_type"],
                           "average_roas": _r(r["roas_p50"], 2), "marginal_roas": _r(r["marginal_roas"], 2),
                           "elasticity": _r(r["elasticity"], 2)}
                          for _, r in marginal_roas(fc, horizon).iterrows()],
        "risks": risks or [],
        "data_quality": {"rows": report.get("total_rows"), "campaigns": report.get("n_campaigns"),
                         "issues": report.get("messages", [])[:6]},
    }

    if anomalies is not None and len(anomalies):
        facts["recent_anomalies"] = [
            {"week_ending": a["week_ending"], "where": f"{a['channel']} {a['campaign_type']}".strip(),
             "metric": a["metric"], "direction": a["direction"], "value": _r(a["value"], 2),
             "typical": _r(a["typical"], 2), "revenue_impact_usd": _r(a["impact_usd"]),
             "week_spend": _r(a["spend"]), "week_revenue": _r(a["revenue"])}
            for _, a in anomalies.head(6).iterrows()]

    if scenario is not None:
        s = scenario[(scenario["horizon_days"] == horizon) & (scenario["level"] == "blended")].iloc[0]
        facts["budget_scenario"] = {
            "planned_spend": _r(s["spend_p50"]), "expected_spend": _r(blended["spend_p50"]),
            "revenue_p10": _r(s["revenue_p10"]), "revenue_p50": _r(s["revenue_p50"]),
            "revenue_p90": _r(s["revenue_p90"]), "roas_p50": _r(s["roas_p50"], 2),
            "revenue_change_vs_baseline_pct": _r((s["revenue_p50"] / max(blended["revenue_p50"], 1e-9) - 1) * 100, 1),
            "spend_change_vs_baseline_pct": _r((s["spend_p50"] / max(blended["spend_p50"], 1e-9) - 1) * 100, 1),
        }

    if allocation is not None and len(allocation):
        gain = allocation["revenue_recommended"].sum() - allocation["revenue_current"].sum()
        facts["budget_optimiser"] = {
            "total_budget": _r(allocation["spend_recommended"].sum()),
            "expected_revenue_gain": _r(gain),
            "expected_revenue_gain_pct": _r(gain / max(allocation["revenue_current"].sum(), 1e-9) * 100, 1),
            "moves": [{"channel": r["channel"], "campaign_type": r["campaign_type"],
                       "from": _r(r["spend_current"]), "to": _r(r["spend_recommended"]),
                       "change_pct": _r(r["change_pct"], 0),
                       "marginal_roas_before": _r(r["marginal_roas_current"], 2),
                       "marginal_roas_after": _r(r["marginal_roas_recommended"], 2)}
                      for _, r in allocation.iterrows() if abs(r["change_pct"]) >= 2],
        }

    bt = pd.DataFrame(bundle.get("backtest", {}).get("by_level", []))
    if len(bt):
        b = bt[(bt["level"] == "blended") & (bt["horizon_days"] == horizon)]
        if len(b):
            b = b.iloc[0]
            facts["model_reliability"] = {
                "backtest_origins": bundle["backtest"].get("n_origins"),
                "blended_wape_pct": _r(b["wape"], 1),
                "blended_wape_if_budget_known_pct": _r(b["wape_budget_known"], 1),
                "interval_coverage_pct": _r(b["coverage"], 1),
                "note": "WAPE = typical size of the miss on a past forecast of this horizon, as % of actual revenue.",
            }
    return facts
