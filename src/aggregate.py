"""Weekly aggregation and feature engineering — v3.1 (All-15-Fixes).

New in v3.1 (All-15-Fixes):
- Fix 5: Cyclical calendar encoding (week_sin, week_cos, month_sin, month_cos)
- Fix 6: Ecommerce event flags (black_friday, cyber_week, christmas, back_to_school,
         jan_slump, valentines_week)
- Fix 7: Budget ratio, log proposed spend, spend saturation, interaction features
         (spend_x_roas, budget_ratio_x_roas, log_spend_x_trend, revenue_trend)
- Fix 8: Target encoding per (channel, campaign_type) — te_revenue, te_roas
         with expanding cumulative mean to prevent data leakage
- Fix 9: Decay-weighted lags (revenue_decay_4w, spend_decay_4w, half-life=2 weeks)
- Fix 10: Channel efficiency features (channel_avg_roas, spend_vs_channel_avg)
- Fix 11: Combined channel imbalance + temporal decay sample weights

Pre-existing differentiators preserved:
- MACD momentum anchor (fast/slow EMA ratio)
- Cross-platform halo features (meta_spend_roll4/8, portfolio_upper_lower_ratio)
- YoY seasonal lags (lag_52, lag_26, yoy_ratio)
- Temporal sample weights (exponential decay)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CHANNEL_WEIGHTS,
    DATA_THRESHOLD_FULL,
    DATA_THRESHOLD_SPARSE,
)


# ─── Aggregation helpers ─────────────────────────────────────────────────────

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily rows to Monday-anchored ISO weeks."""
    tmp = df.copy()
    tmp["week"] = tmp["date"].dt.to_period("W-MON").dt.start_time

    numeric_sum = ["spend", "revenue", "clicks", "impressions", "conversions"]
    numeric_mean = ["daily_budget"]

    agg_spec: dict[str, str | object] = {
        **{c: "sum" for c in numeric_sum if c in tmp.columns},
        **{c: "mean" for c in numeric_mean if c in tmp.columns},
        "campaign_name":       "first",
        "campaign_type":       "first",
        "audience_segment":    "first",
        "channel":             "first",
    }
    for flag in ["flag_zero_spend_nonzero_revenue", "flag_zero_revenue_nonzero_spend"]:
        if flag in tmp.columns:
            agg_spec[flag] = "any"

    weekly = (
        tmp.groupby(["channel", "campaign_id", "week"], as_index=False)
        .agg(agg_spec)
        .rename(columns={"week": "date"})
    )
    return weekly


def fill_missing_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every group has a contiguous weekly time-series (fill gaps with zeros).
    
    Missing weeks are filled with zero spend/revenue so that the model sees
    inactive weeks rather than non-existent rows — prevents lag features from
    accidentally skipping over periods of inactivity.
    """
    if df.empty:
        return df

    group_cols = ["channel", "campaign_id"]
    missing_cols = [c for c in group_cols if c not in df.columns]
    if missing_cols:
        return df  # Skip if grouping cols not available

    min_date = df["date"].min()
    max_date = df["date"].max()
    all_weeks = pd.date_range(start=min_date, end=max_date, freq="W-MON")

    records = []
    for (ch, cid), grp in df.groupby(group_cols):
        meta = grp.iloc[0]
        existing_dates = set(grp["date"].dt.normalize())
        for week in all_weeks:
            wn = week.normalize()
            if wn not in existing_dates:
                filler = {
                    "channel": ch,
                    "campaign_id": cid,
                    "date": week,
                    "spend": 0.0,
                    "revenue": 0.0,
                    "clicks": 0.0,
                    "impressions": 0.0,
                    "conversions": 0.0,
                    "daily_budget": float(meta.get("daily_budget", 0) or 0),
                    "campaign_name": meta.get("campaign_name", ""),
                    "campaign_type": meta.get("campaign_type", "UNKNOWN"),
                    "audience_segment": meta.get("audience_segment", "Generic"),
                }
                records.append(filler)

    if records:
        filled = pd.concat([df, pd.DataFrame(records)], ignore_index=True)
        filled = filled.sort_values(group_cols + ["date"]).reset_index(drop=True)
        return filled
    return df


def to_daily_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Daily-level aggregation (used for ablation only)."""
    tmp = df.copy()
    tmp["date"] = tmp["date"].dt.normalize()
    numeric_sum = ["spend", "revenue", "clicks", "impressions", "conversions"]
    agg_spec: dict[str, str] = {
        **{c: "sum" for c in numeric_sum if c in tmp.columns},
        "daily_budget":     "mean",
        "campaign_name":    "first",
        "campaign_type":    "first",
        "audience_segment": "first",
        "channel":          "first",
    }
    daily = tmp.groupby(["channel", "campaign_id", "date"], as_index=False).agg(agg_spec)
    daily["day_of_week"] = daily["date"].dt.dayofweek
    return daily


# ─── Group-key helpers ───────────────────────────────────────────────────────

def _group_keys(panel: pd.DataFrame, granularity: str = "auto") -> list[str]:
    if granularity == "campaign":
        return ["channel", "campaign_id"]
    if granularity == "campaign_type":
        return ["channel", "campaign_type"]
    if granularity == "channel":
        return ["channel"]
    # auto
    if "campaign_id" in panel.columns:
        return ["channel", "campaign_id"]
    if "campaign_type" in panel.columns and panel["campaign_type"].nunique() > 1:
        return ["channel", "campaign_type"]
    return ["channel"]


# ─── Cross-platform halo features ────────────────────────────────────────────

def add_cross_platform_halo(panel: pd.DataFrame) -> pd.DataFrame:
    """Add Meta spend signals as halo features for Google/Bing search models.

    Blueprint Implementation: Cross-Platform Interactive Lags
    Builds the pivot safely to prevent target leakage.
    """
    out = panel.copy()
    if "channel" not in out.columns or "date" not in out.columns:
        out["meta_spend_roll4"] = 0.0
        out["meta_spend_roll8"] = 0.0
        out["portfolio_upper_lower_ratio"] = 1.0
        return out

    # 1. Build Cross-Platform Pivot Matrix safely to prevent target leakage
    platform_pivot = out.pivot_table(
        index='date', 
        columns='channel', 
        values='spend', 
        aggfunc='sum'
    ).fillna(0)
    
    # Crucial: Shift by 1 or 2 weeks to avoid lookahead leakage
    halo_1w = platform_pivot.shift(1).add_prefix('halo_lag_1w_')
    halo_2w = platform_pivot.shift(2).add_prefix('halo_lag_2w_')
    
    halo_df = halo_1w.join(halo_2w)
    out = out.join(halo_df, on='date')
    
    # Backward compatibility with v3.1 features (meta_spend_roll4 / 8)
    if "meta" in platform_pivot.columns:
        meta_shifted = platform_pivot['meta'].shift(1)
        out["meta_spend_roll4"] = out["date"].map(meta_shifted.rolling(4, min_periods=1).mean()).fillna(0.0)
        out["meta_spend_roll8"] = out["date"].map(meta_shifted.rolling(8, min_periods=1).mean()).fillna(0.0)
    else:
        out["meta_spend_roll4"] = 0.0
        out["meta_spend_roll8"] = 0.0
        
    for col in halo_df.columns:
        out[col] = out[col].fillna(0.0)

    # ── Portfolio: upper-funnel (Prospecting) vs lower-funnel (Retargeting) ──
    prospecting_types = {"PROSPECTING", "BRAND", "VIDEO", "DISPLAY", "DEMAND_GEN"}
    retargeting_types = {"REMARKETING", "SEARCH", "PERFORMANCE_MAX", "SHOPPING"}

    if "campaign_type" in out.columns:
        weekly_portfolio = (
            out.groupby("date")
            .apply(lambda g: pd.Series({
                "prosp_spend": g.loc[g["campaign_type"].isin(prospecting_types), "spend"].sum(),
                "retary_spend": g.loc[g["campaign_type"].isin(retargeting_types), "spend"].sum(),
            }))
            .reset_index()
        )
        # Shift 1 to prevent leakage
        weekly_portfolio["portfolio_upper_lower_ratio"] = (
            weekly_portfolio["prosp_spend"].shift(1) / (weekly_portfolio["retary_spend"].shift(1) + 1.0)
        ).clip(0, 20)
        out = out.merge(weekly_portfolio[["date", "portfolio_upper_lower_ratio"]], on="date", how="left")
    else:
        out["portfolio_upper_lower_ratio"] = 1.0

    out["portfolio_upper_lower_ratio"] = out["portfolio_upper_lower_ratio"].fillna(1.0)
    return out


# ─── Temporal + channel sample weights (Fix 11) ──────────────────────────────

def compute_temporal_weights(
    panel: pd.DataFrame,
    decay: float = 0.985,
    apply_channel_weights: bool = True,
) -> np.ndarray:
    """Combined exponential temporal decay + channel imbalance correction (Fix 11).
    
    Fix 11: Google has ~19k rows, Bing ~2.8k. Without channel weights the model
    ignores Bing. This combines both corrections for strictly better results than
    either alone.
    
    decay=0.985 over ~105 weeks → 0.985^105 ≈ 0.20 (recent=1.0, oldest≈0.2).
    Channel multipliers: google=1.0, meta=2.0, bing=3.0.
    """
    dates = pd.to_datetime(panel["date"])
    max_date = dates.max()
    weeks_ago = ((max_date - dates).dt.days / 7.0).values
    temporal = np.power(decay, weeks_ago).clip(0.2, 1.0)

    if apply_channel_weights and "channel" in panel.columns:
        # channel may be integer-encoded after _encode_categories() in model.py
        # Convert to str first to safely call .str.lower() and map
        ch_series = panel["channel"].astype(str).str.lower()
        ch_weights = ch_series.map(CHANNEL_WEIGHTS).fillna(1.0).values
        combined = temporal * ch_weights
    else:
        combined = temporal

    return combined.astype(np.float32)


# ─── Fix 5: Cyclical calendar encoding ───────────────────────────────────────

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fix 5: Cyclical sin/cos encoding for week_of_year and month.
    
    Week 53 and week 1 are adjacent in reality — ordinal encoding treats them as
    maximally different. Cyclical encoding wraps the calendar correctly.
    """
    out = df.copy()
    if "week_of_year" not in out.columns:
        out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    if "month" not in out.columns:
        out["month"] = out["date"].dt.month

    out["week_sin"] = np.sin(2 * np.pi * out["week_of_year"] / 53)
    out["week_cos"] = np.cos(2 * np.pi * out["week_of_year"] / 53)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


# ─── Fix 6: Ecommerce event flags ────────────────────────────────────────────

def add_ecommerce_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Fix 6: Specific high-revenue ecommerce event flags.
    
    Black Friday and Cyber Week are the highest-revenue weeks of the year for
    ecommerce. Without these flags the model massively under-predicts those weeks.
    These are NOT nice-to-have — they are the weeks that determine MAE quality.
    """
    out = df.copy()

    # Ensure date columns
    dt = pd.to_datetime(out["date"])
    woy = dt.dt.isocalendar().week.astype(int)
    month = dt.dt.month
    day = dt.dt.day

    # High-revenue ecommerce events
    out["is_black_friday_week"] = ((woy == 47) & (month == 11)).astype(int)
    out["is_cyber_week"]        = ((woy == 48) & (month == 11)).astype(int)
    out["is_christmas_week"]    = ((month == 12) & day.between(22, 28)).astype(int)
    out["is_back_to_school"]    = ((month == 8) | ((month == 7) & (day >= 20))).astype(int)
    out["is_jan_slump"]         = (month == 1).astype(int)
    out["is_valentines_week"]   = ((month == 2) & day.between(7, 14)).astype(int)

    # is_q4 is already computed in add_time_features — no duplicate here
    return out


# ─── Fix 7: Budget ratio & interaction features ───────────────────────────────

def add_budget_and_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fix 7: Budget ratio, log spend, spend saturation, and interaction features.
    
    Friend's feature importance shows:
      #1: proposed_weekly_spend
      #2: spend_saturation
      #3: budget_ratio
    These three plus interaction terms are the most impactful missing features.
    """
    out = df.copy()
    eps = 1e-9

    # Budget ratio — how current planned spend compares to recent historical average
    spend_roll_hist = out.get("spend_roll_4", out.get("spend_rolling_mean_4", out["spend"].clip(lower=eps)))
    out["budget_ratio"] = (out["planned_spend"].clip(lower=0) / (spend_roll_hist.replace(0, np.nan).fillna(eps))).clip(0, 20)

    # Log-compressed planned spend (diminishing returns on the feature itself)
    out["log_proposed_spend"] = np.log1p(out["planned_spend"].clip(lower=0))

    # Spend saturation based on Google Lightweight MMM research
    if "channel" in out.columns:
        median_channel_spend = out.groupby("channel")["planned_spend"].transform("median").replace(0, np.nan).fillna(eps)
        out["spend_saturation"] = np.log1p(out["planned_spend"].clip(lower=0) / median_channel_spend)
    else:
        global_median = out["planned_spend"].median()
        out["spend_saturation"] = np.log1p(out["planned_spend"].clip(lower=0) / max(global_median, eps))

    # Revenue trend — short vs long momentum (needed for interaction)
    roll4 = out.get("roll_mean_4", out["revenue"].rolling(4, min_periods=1).mean())
    roll8 = out.get("roll_mean_8", out["revenue"].rolling(8, min_periods=1).mean())
    out["revenue_trend"] = (roll4 / (roll8.replace(0, np.nan).fillna(eps))).clip(0, 5)

    # Interaction features
    roas_lag = out.get("roas_lag_1", out.get("hist_roas_1", pd.Series(0.0, index=out.index)))
    roas_roll = out.get("roas_rolling_mean_4", out.get("hist_roas_roll4", pd.Series(0.0, index=out.index)))

    out["spend_x_roas"]        = (spend_roll_hist * roas_roll).clip(0, 1e8)
    out["budget_ratio_x_roas"] = (out["budget_ratio"] * roas_lag).clip(0, 100)
    out["log_spend_x_trend"]   = (out["log_proposed_spend"] * out["revenue_trend"]).clip(0, 100)

    return out


# ─── Fix 8: Target encoding ───────────────────────────────────────────────────

def add_target_encoding(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    target_col: str = "revenue",
) -> pd.DataFrame:
    """Fix 8: Leave-one-out expanding cumulative mean — no data leakage.
    
    For each row, uses only data from weeks BEFORE that row (expanding mean shifted
    by 1 week). First row of each group gets global mean as prior.
    """
    out = df.copy().sort_values("date")
    group_cols = group_cols or ["channel", "campaign_type"]

    # Only encode if both group columns exist
    available_cols = [c for c in group_cols if c in out.columns]
    if len(available_cols) < 2:
        # Fall back to single column or skip
        available_cols = [c for c in ["channel"] if c in out.columns]
    if not available_cols:
        out["te_revenue"] = out[target_col].mean()
        out["te_roas"] = 0.0
        return out

    global_rev_mean = out[target_col].mean()

    # Expanding mean shifted by 1 to prevent lookahead
    out["te_revenue"] = (
        out.groupby(available_cols)[target_col]
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(global_rev_mean)
    )

    # ROAS target encoding
    if "spend" in out.columns:
        out["_roas_tmp"] = (out[target_col] / out["spend"].replace(0, np.nan)).fillna(0).clip(0, 50)
        global_roas_mean = out["_roas_tmp"].mean()
        out["te_roas"] = (
            out.groupby(available_cols)["_roas_tmp"]
            .transform(lambda x: x.expanding().mean().shift(1))
            .fillna(global_roas_mean)
        )
        out = out.drop(columns=["_roas_tmp"])
    else:
        out["te_roas"] = 0.0

    return out


# ─── Fix 9: Decay-weighted lags ───────────────────────────────────────────────

def add_decay_weighted_lags(
    df: pd.DataFrame,
    group_keys: list[str] | None = None,
    target_col: str = "revenue",
    halflife: float = 2.0,
) -> pd.DataFrame:
    """Fix 9: Exponentially decay-weighted sum of last 4 weeks (halflife=2 weeks).
    
    Weights: [1.0, 0.71, 0.50, 0.35] normalised (most recent first).
    More sensitive to recent budget changes than simple rolling mean.
    """
    out = df.copy()
    weights_raw = np.array([0.5 ** (i / halflife) for i in range(4)])
    weights = weights_raw / weights_raw.sum()  # normalise

    gk = group_keys or _group_keys(df)
    # Ensure all group keys exist
    gk = [k for k in gk if k in df.columns]

    def _decay_weighted_mean(x: pd.Series) -> pd.Series:
        result = pd.Series(np.nan, index=x.index, dtype=float)
        x_vals = x.values
        for i in range(len(x_vals)):
            if i < 4:
                result.iloc[i] = np.nan
            else:
                # Last 4 values before current index (shifted by 1 for no leakage)
                past = x_vals[max(0, i - 4):i][::-1]  # most recent first
                if len(past) == 4:
                    result.iloc[i] = float(np.dot(past, weights))
                elif len(past) > 0:
                    w_sub = weights_raw[:len(past)]
                    w_sub = w_sub / w_sub.sum()
                    result.iloc[i] = float(np.dot(past, w_sub))
        return result

    if gk:
        out["revenue_decay_4w"] = out.groupby(gk)[target_col].transform(_decay_weighted_mean)
        if "spend" in out.columns:
            out["spend_decay_4w"] = out.groupby(gk)["spend"].transform(_decay_weighted_mean)
        else:
            out["spend_decay_4w"] = 0.0
    else:
        out["revenue_decay_4w"] = out[target_col].transform(_decay_weighted_mean)
        out["spend_decay_4w"] = 0.0

    out["revenue_decay_4w"] = out["revenue_decay_4w"].fillna(0.0)
    out["spend_decay_4w"] = out["spend_decay_4w"].fillna(0.0)
    return out


# ─── Fix 10: Channel efficiency features ─────────────────────────────────────

def add_channel_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fix 10: Relative ROAS efficiency vs channel average.
    
    A campaign earning 3x ROAS when channel average is 5x is underperforming.
    The same 3x when channel average is 2x is a star. Relative efficiency is
    more informative than absolute ROAS.
    """
    out = df.copy()

    if "channel" not in out.columns:
        out["channel_avg_roas"] = 0.0
        out["spend_vs_channel_avg"] = 1.0
        return out

    roas_col = out.get("hist_roas_roll4", out.get("roas_rolling_mean_4", None))
    if roas_col is None:
        # Compute on the fly from revenue and spend
        spend_safe = out["spend"].replace(0, np.nan)
        roas_col = (out["revenue"] / spend_safe).fillna(0).clip(0, 50)

    channel_avg_roas = out.groupby("channel")[roas_col.name if hasattr(roas_col, "name") else "revenue"].transform("mean")
    out["channel_avg_roas"] = channel_avg_roas.fillna(0.0)
    out["spend_vs_channel_avg"] = (roas_col / (channel_avg_roas + 1e-9)).clip(0, 10).fillna(1.0)

    return out


# ─── Core feature engineering ────────────────────────────────────────────────

def add_time_features(panel: pd.DataFrame, freq: str = "weekly") -> pd.DataFrame:
    """Compute all temporal, ratio, maturity, and dual-engine target features.
    
    v3.1: Includes all new features from Fixes 5-10.
    """
    out = panel.copy()
    group_keys = _group_keys(out)
    out = out.sort_values(group_keys + ["date"])

    # ── Calendar features ────────────────────────────────────────────────────
    out["week_of_year"]   = out["date"].dt.isocalendar().week.astype(int)
    out["month"]          = out["date"].dt.month
    out["quarter"]        = out["date"].dt.quarter
    out["year"]           = out["date"].dt.year
    out["is_q4"]          = (out["month"] >= 10).astype(int)
    out["is_holiday_adj"] = out["week_of_year"].isin([48, 49, 50, 51, 52, 1]).astype(int)

    # Fix 5: Cyclical calendar encoding
    out = add_cyclical_features(out)

    # Fix 6: Ecommerce event flags
    out = add_ecommerce_event_flags(out)

    # ── Global Portfolio Momentum ────────────────────────────────────────────
    if "date" in out.columns:
        global_weekly = out.groupby("date")["revenue"].sum().reset_index(name="global_revenue")
        global_weekly["global_portfolio_wow_change"] = (
            global_weekly["global_revenue"].shift(1) / (global_weekly["global_revenue"].shift(2) + 1.0)
        ).clip(0, 5) - 1.0
        out = out.merge(global_weekly[["date", "global_portfolio_wow_change"]], on="date", how="left")
        out["global_portfolio_wow_change"] = out["global_portfolio_wow_change"].fillna(0.0)

    first_date = out.groupby(group_keys, dropna=False)["date"].transform("min")
    out["weeks_since_start"] = ((out["date"] - first_date).dt.days // 7).astype(int)
    out["days_since_start"]  = (out["date"] - first_date).dt.days
    out["campaign_age_weeks"] = out["days_since_start"] / 7.0

    # ── Revenue lags ─────────────────────────────────────────────────────────
    rev = out.groupby(group_keys)["revenue"]
    for lag in [1, 2, 4, 8, 12]:
        out[f"lag_{lag}"] = rev.shift(lag)

    # ── Year-over-year lags (seasonal signal) ────────────────────────────────
    out["log_revenue_tmp"] = np.log1p(out["revenue"].clip(lower=0))
    lrev_grp = out.groupby(group_keys)["log_revenue_tmp"]
    out["lrev_lag_52"] = lrev_grp.shift(52)
    out["lrev_lag_26"] = lrev_grp.shift(26)

    out["yoy_ratio"] = (
        (rev.shift(1) / (rev.shift(52) + 1))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(0, 20)
    )

    out["yoy_roll4_lag52"] = lrev_grp.transform(
        lambda x: x.shift(50).rolling(4, min_periods=2).mean()
    )

    out = out.drop(columns=["log_revenue_tmp"])

    # ── Rolling revenue stats ────────────────────────────────────────────────
    def _roll(s: pd.Series, window: int) -> pd.Series:
        return s.shift(1).rolling(window, min_periods=1)

    for w in [4, 8]:
        out[f"roll_mean_{w}"] = rev.transform(lambda s: _roll(s, w).mean())
        if w == 4:
            out["roll_std_4"] = rev.transform(lambda s: _roll(s, 4).std())

    safe_mean = out["roll_mean_4"].replace(0, np.nan)
    out["roll_cv_4"] = (out["roll_std_4"] / safe_mean).fillna(0).clip(0, 10)

    # Backward-compatible aliases
    out["rolling_mean_4"] = out["roll_mean_4"]
    out["rolling_std_4"]  = out["roll_std_4"]

    # ── Spend lags + rolling ─────────────────────────────────────────────────
    sp = out.groupby(group_keys)["spend"]
    out["spend_lag_1"]     = sp.shift(1)
    out["spend_roll_4"]    = sp.transform(lambda s: _roll(s, 4).mean())
    out["spend_lag_2"]     = sp.shift(2)

    out["spend_rolling_mean_4"] = out["spend_roll_4"]

    spend_prev      = sp.shift(1)
    spend_prev_prev = sp.shift(2)
    out["spend_change_ratio"] = (
        (spend_prev / spend_prev_prev.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    out["spend_ratio_vs_hist"] = (
        (out["spend"] / out["spend_roll_4"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(0, 10)
    )

    # ── ROAS / efficiency ratio features ─────────────────────────────────────
    safe_sp1 = out["spend_lag_1"].replace(0, np.nan)
    out["roas_lag_1"]  = (out["lag_1"] / safe_sp1).fillna(0).clip(0, 100)
    out["hist_roas_1"] = out["roas_lag_1"]

    safe_sp_roll = out["spend_roll_4"].replace(0, np.nan)
    out["hist_roas_roll4"] = (out["roll_mean_4"] / safe_sp_roll).fillna(0).clip(0, 100)
    out["roas_rolling_mean_4"] = out["hist_roas_roll4"]

    clicks_roll4 = out.groupby(group_keys)["clicks"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).sum()
    )
    rev_roll4_sum = rev.transform(lambda s: s.shift(1).rolling(4, min_periods=1).sum())
    out["rpc_roll4"] = (
        (rev_roll4_sum / (clicks_roll4 + 1))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ── Action 5 / Step 4: Brand ROAS stability feature ──
    def _cv_4w(x):
        return x.rolling(4, min_periods=2).std() / (x.rolling(4, min_periods=2).mean() + 1e-9)
    
    out['brand_roas_stability'] = (
        out.groupby(group_keys)['hist_roas_1']
           .transform(_cv_4w)
           .fillna(0.0)
    )

    is_brand_segment = False
    if "audience_segment" in out.columns:
        is_brand_segment = out['audience_segment'].astype(str).str.upper().isin(['TM', 'BRAND'])
    elif "campaign_type" in out.columns:
        from src.config import BRAND_CAMPAIGN_TYPES
        is_brand_segment = out["campaign_type"].isin(BRAND_CAMPAIGN_TYPES)

    out['is_tm_brand'] = is_brand_segment.astype(int)
    out['brand_instability_signal'] = out['is_tm_brand'] * out['brand_roas_stability']

    # ── Campaign maturity ─────────────────────────────────────────────────────
    out["weeks_active"]    = out.groupby(group_keys).cumcount().astype(int)
    out["is_new_campaign"] = (out["weeks_active"] < 4).astype(int)

    # ── Zero-inflation features ───────────────────────────────────────────────
    out["recent_zero_rate"] = rev.transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).apply(lambda w: (w == 0).mean(), raw=True)
    )
    out["last_week_zero"] = (rev.shift(1) == 0).astype(int)
    out["consecutive_zeros"] = rev.transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).apply(lambda w: (w == 0).sum(), raw=True)
    )

    # ── Direct Horizon Multi-Step Targets (30, 60, 90 days) ───────────────────
    def forward_sum(s: pd.Series, window: int) -> pd.Series:
        return s.iloc[::-1].rolling(window, min_periods=window).sum().shift(1).iloc[::-1]

    out["target_30"] = rev.transform(lambda s: forward_sum(s, 4))
    out["target_60"] = rev.transform(lambda s: forward_sum(s, 8))
    out["target_90"] = rev.transform(lambda s: forward_sum(s, 13))

    sp_grp = out.groupby(group_keys)["spend"]
    out["planned_spend_30"] = sp_grp.transform(lambda s: forward_sum(s, 4))
    out["planned_spend_60"] = sp_grp.transform(lambda s: forward_sum(s, 8))
    out["planned_spend_90"] = sp_grp.transform(lambda s: forward_sum(s, 13))

    # ── MACD Anchor (existing differentiator — preserved) ────────────────────
    ema_fast = rev.transform(lambda s: s.shift(1).ewm(span=2, adjust=False).mean()).fillna(0)
    ema_slow = rev.transform(lambda s: s.shift(1).ewm(span=8, adjust=False).mean()).fillna(0)
    macd_ratio = (ema_fast / (ema_slow + 1.0)).clip(0.5, 1.5)
    out["ema_revenue"] = ema_slow * macd_ratio

    out["anchor_30"] = out["ema_revenue"] * 4 * out["yoy_ratio"]
    out["anchor_60"] = out["ema_revenue"] * 8 * out["yoy_ratio"]
    out["anchor_90"] = out["ema_revenue"] * 13 * out["yoy_ratio"]

    out["residual_30"] = out["target_30"] - out["anchor_30"]
    out["residual_60"] = out["target_60"] - out["anchor_60"]
    out["residual_90"] = out["target_90"] - out["anchor_90"]

    # ── Log-scale columns ─────────────────────────────────────────────────────
    out["log_revenue"] = np.log1p(out["revenue"].clip(lower=0))
    out["log_spend"]   = np.log1p(out["spend"].clip(lower=0))

    # ── Planned spend features (needed for Fix 7 interaction terms) ───────────
    # Use 30-day planned spend as the default "planned_spend" before training sets it
    out["planned_spend"] = out["planned_spend_30"].fillna(0.0)

    # Fix 7: Budget ratio & interaction features
    out = add_budget_and_interaction_features(out)

    # Fix 9: Decay-weighted lags
    out = add_decay_weighted_lags(out, group_keys=group_keys)

    # Fix 10: Channel efficiency features
    out = add_channel_efficiency_features(out)

    if freq == "daily":
        out["day_of_week"] = out["date"].dt.dayofweek

    return out


# ─── Data sufficiency classification ────────────────────────────────────────

def classify_group_sufficiency(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `data_tier` column per (channel, campaign_type): full / sparse / fallback."""
    out = panel.copy()
    keys = ["channel", "campaign_type"] if "campaign_type" in out.columns else ["channel"]

    def _tier(g: pd.DataFrame) -> str:
        nonzero_weeks = int((g["revenue"] > 0).sum())
        if nonzero_weeks >= DATA_THRESHOLD_FULL:
            return "full"
        if nonzero_weeks >= DATA_THRESHOLD_SPARSE:
            return "sparse"
        return "fallback"

    tier_map = out.groupby(keys).apply(_tier).reset_index()
    tier_map.columns = keys + ["data_tier"]  # type: ignore[assignment]
    out = out.merge(tier_map, on=keys, how="left")
    out["data_tier"] = out["data_tier"].fillna("fallback")
    return out


# ─── Build training panel ────────────────────────────────────────────────────

_YOY_DEFAULTS = {
    "lrev_lag_52": 0.0, "lrev_lag_26": 0.0,
    "yoy_ratio": 1.0, "yoy_roll4_lag52": 0.0,
}
_ZERO_DEFAULTS = {
    "recent_zero_rate": 0.0, "last_week_zero": 0, "consecutive_zeros": 0.0,
}
_HALO_DEFAULTS = {
    "meta_spend_roll4": 0.0, "meta_spend_roll8": 0.0,
    "portfolio_upper_lower_ratio": 1.0,
}
# Fix 5: Cyclical calendar defaults
_CYCLICAL_DEFAULTS = {
    "week_sin": 0.0, "week_cos": 1.0,
    "month_sin": 0.0, "month_cos": 1.0,
}
# Fix 6: Ecommerce event defaults
_EVENT_DEFAULTS = {
    "is_black_friday_week": 0, "is_cyber_week": 0,
    "is_christmas_week": 0, "is_back_to_school": 0,
    "is_jan_slump": 0, "is_valentines_week": 0,
}
# Fix 7-10: New feature defaults
_NEW_FEATURE_DEFAULTS = {
    "budget_ratio": 1.0, "log_proposed_spend": 0.0, "spend_saturation": 0.0,
    "spend_x_roas": 0.0, "budget_ratio_x_roas": 0.0, "log_spend_x_trend": 0.0,
    "revenue_trend": 1.0, "te_revenue": 0.0, "te_roas": 0.0,
    "channel_avg_roas": 0.0, "spend_vs_channel_avg": 1.0,
    "revenue_decay_4w": 0.0, "spend_decay_4w": 0.0,
}

FILL_DEFAULTS: dict[str, float] = {
    "lag_2": 0.0, "lag_4": 0.0, "lag_8": 0.0, "lag_12": 0.0,
    "roll_mean_4": 0.0, "roll_mean_8": 0.0, "roll_std_4": 0.0, "roll_cv_4": 0.0,
    "rolling_std_4": 0.0, "roas_rolling_mean_4": 0.0,
    "hist_roas_1": 0.0, "hist_roas_roll4": 0.0, "rpc_roll4": 0.0,
    "spend_lag_1": 0.0, "spend_lag_2": 0.0, "spend_roll_4": 0.0,
    "spend_rolling_mean_4": 0.0, "spend_ratio_vs_hist": 1.0,
    "spend_change_ratio": 1.0, "roas_lag_1": 0.0,
    "campaign_age_weeks": 0.0, "days_since_start": 0.0,
    "weeks_active": 0, "is_new_campaign": 0,
    "is_q4": 0, "is_holiday_adj": 0, "weeks_since_start": 0,
    "target_30": 0.0, "target_60": 0.0, "target_90": 0.0,
    "planned_spend_30": 0.0, "planned_spend_60": 0.0, "planned_spend_90": 0.0,
    "anchor_30": 0.0, "anchor_60": 0.0, "anchor_90": 0.0,
    "residual_30": 0.0, "residual_60": 0.0, "residual_90": 0.0,
    "global_portfolio_wow_change": 0.0,
    **_YOY_DEFAULTS,
    **_ZERO_DEFAULTS,
    **_HALO_DEFAULTS,
    **_CYCLICAL_DEFAULTS,
    **_EVENT_DEFAULTS,
    **_NEW_FEATURE_DEFAULTS,
}


def build_training_panel(df: pd.DataFrame, freq: str = "weekly") -> pd.DataFrame:
    """Full pipeline: weekly agg → fill gaps → halo features → time features → classify → target encode."""
    if freq == "weekly":
        panel = to_weekly(df)
        panel = fill_missing_weeks(panel)
    else:
        panel = to_daily_agg(df)

    # Cross-platform halo features (computed before per-group features)
    panel = add_cross_platform_halo(panel)

    panel = add_time_features(panel, freq=freq)
    panel = classify_group_sufficiency(panel)

    # Fix 8: Target encoding — applied after all other features, before train/test split
    # This uses expanding mean on the full panel (safe because we shift by 1 week)
    panel = add_target_encoding(panel)

    # Drop rows where lag_1 is NaN (first week of each group)
    panel = panel.dropna(subset=["lag_1"])

    panel = panel.fillna(FILL_DEFAULTS)
    return panel


# ─── Level aggregation (campaign_type / channel) ─────────────────────────────

def aggregate_by_level(panel: pd.DataFrame, level: str) -> pd.DataFrame:
    """Re-aggregate weekly campaign-level panel to campaign_type or channel level."""
    if level == "campaign":
        return panel.copy()

    group_map = {
        "campaign_type": ["channel", "campaign_type", "date"],
        "channel":       ["channel", "date"],
    }
    keys = group_map[level]
    numeric = ["spend", "revenue", "clicks", "impressions", "conversions", "daily_budget"]
    agg: dict[str, str] = {c: "sum" for c in numeric if c in panel.columns}
    if "audience_segment" in panel.columns and "audience_segment" not in keys:
        agg["audience_segment"] = "first"

    grouped = panel.groupby(keys, as_index=False).agg(agg)

    # Add halo features at group level (take mean across campaigns in group)
    for halo_col in ["meta_spend_roll4", "meta_spend_roll8", "portfolio_upper_lower_ratio"]:
        if halo_col in panel.columns:
            halo_agg = panel.groupby(keys)[halo_col].mean().reset_index()
            grouped = grouped.merge(halo_agg, on=keys, how="left")

    grouped = add_time_features(grouped, freq="weekly")
    grouped = classify_group_sufficiency(grouped)

    # Fix 8: Target encoding at aggregated level
    grouped = add_target_encoding(grouped)

    level_fill = {**FILL_DEFAULTS}
    grouped = grouped.fillna(level_fill)
    return grouped
