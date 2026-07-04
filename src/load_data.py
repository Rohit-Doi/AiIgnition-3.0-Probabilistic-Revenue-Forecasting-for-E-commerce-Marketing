"""Load and harmonize multi-channel campaign CSVs into a canonical schema.

Changes vs v1:
- Future-date filtering
- Strict deduplication on (channel, campaign_id, date)
- spend/revenue clipped to ≥0 with null counts logged
- Anomaly flags for zero-spend/non-zero-revenue rows
- Richer audience-segment parsing (brand/nonbrand/prospecting/remarketing/dpa)
- Richer Meta campaign_type inference (REMARKETING/PROSPECTING/DISPLAY/VIDEO/BRAND)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import META_CONVERSION_AS_REVENUE

CANONICAL_COLUMNS = [
    "date",
    "channel",
    "campaign_id",
    "campaign_name",
    "campaign_type",
    "audience_segment",
    "spend",
    "revenue",
    "clicks",
    "impressions",
    "conversions",
    "daily_budget",
    # Anomaly flags (kept throughout pipeline; excluded from model features)
    "flag_zero_spend_nonzero_revenue",
    "flag_zero_revenue_nonzero_spend",
]


# ─── Audience segment parsing ────────────────────────────────────────────────

def _parse_audience_segment(name: str) -> str:
    """Parse audience segment from campaign name with expanded logic."""
    if not isinstance(name, str):
        return "Generic"
    lower = name.lower()
    # Brand / Trade-marked terms
    if any(tok in lower for tok in ["_tm_", " tm ", "_tm", "tm_", "brand", "branded"]):
        return "brand"
    # Non-brand / generic paid search
    if any(tok in lower for tok in ["_ntm_", "ntm_", " ntm ", "nonbrand", "non_brand", "generic"]):
        return "nonbrand"
    # Prospecting (top-of-funnel)
    if "prospecting" in lower or "prosp" in lower:
        return "prospecting"
    # Remarketing / retargeting / DPA
    if "remarketing" in lower or "retarg" in lower or "retargeting" in lower:
        return "remarketing"
    if "dpa" in lower or "dynamic" in lower:
        return "dpa"
    return "Generic"


# ─── Campaign type normalization ─────────────────────────────────────────────

_TYPE_MAP: dict[str, str] = {
    "SEARCH": "SEARCH",
    "PERFORMANCE_MAX": "PERFORMANCE_MAX",
    "PERFORMANCEMAX": "PERFORMANCE_MAX",
    "PMAX": "PERFORMANCE_MAX",
    "DISPLAY": "DISPLAY",
    "VIDEO": "VIDEO",
    "DEMAND_GEN": "DEMAND_GEN",
    "DEMANDGEN": "DEMAND_GEN",
    "SHOPPING": "SHOPPING",
    # Bing equivalents
    "AUDIENCE": "DISPLAY",
    # Explicit meta types
    "REMARKETING": "REMARKETING",
    "PROSPECTING": "PROSPECTING",
    "BRAND": "BRAND",
}


def _normalize_campaign_type(raw: str, channel: str = "") -> str:
    if not isinstance(raw, str):
        return "UNKNOWN"
    val = raw.strip().upper().replace(" ", "_").replace("-", "_")
    return _TYPE_MAP.get(val, "UNKNOWN")


def _infer_meta_campaign_type(name: str) -> str:
    """Infer Meta campaign type from campaign name prefix / keywords."""
    if not isinstance(name, str):
        return "DISPLAY"
    lower = name.lower()
    if "video" in lower:
        return "VIDEO"
    if any(tok in lower for tok in ["brand", "_tm_", " tm ", "tm_"]):
        return "BRAND"
    if "remarketing" in lower or "retarg" in lower or "dpa" in lower:
        return "REMARKETING"
    if "prospecting" in lower or "prosp" in lower:
        return "PROSPECTING"
    # Default: Meta is mostly display-type placements
    return "DISPLAY"


# ─── Per-channel loaders ─────────────────────────────────────────────────────

def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean anomaly flags and clip spend/revenue to ≥ 0."""
    df = df.copy()
    # Clip before flagging so comparison is clean
    null_spend  = int(df["spend"].isna().sum())
    null_rev    = int(df["revenue"].isna().sum())
    neg_spend   = int((df["spend"].fillna(0) < 0).sum())
    neg_rev     = int((df["revenue"].fillna(0) < 0).sum())

    df["spend"]   = df["spend"].clip(lower=0).fillna(0)
    df["revenue"] = df["revenue"].clip(lower=0).fillna(0)

    df["flag_zero_spend_nonzero_revenue"] = (
        (df["spend"] == 0) & (df["revenue"] > 0)
    ).astype(bool)
    df["flag_zero_revenue_nonzero_spend"] = (
        (df["revenue"] == 0) & (df["spend"] > 0)
    ).astype(bool)

    # Surface counts for validation report (stored as attrs)
    df.attrs.setdefault("load_stats", {})
    ch = df["channel"].iloc[0] if len(df) else "unknown"
    df.attrs["load_stats"].update({
        f"{ch}_null_spend": null_spend,
        f"{ch}_null_revenue": null_rev,
        f"{ch}_neg_spend": neg_spend,
        f"{ch}_neg_revenue": neg_rev,
    })
    return df


def load_google(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({
        "date":           pd.to_datetime(df["segments_date"], errors="coerce"),
        "channel":        "google",
        "campaign_id":    df["campaign_id"].astype(str),
        "campaign_name":  df["campaign_name"],
        "campaign_type":  df["campaign_advertising_channel_type"].map(
                              lambda x: _normalize_campaign_type(x, "google")),
        "audience_segment": df["campaign_name"].map(_parse_audience_segment),
        "spend":          pd.to_numeric(df["metrics_cost_micros"], errors="coerce") / 1_000_000,
        "revenue":        pd.to_numeric(df["metrics_conversions_value"], errors="coerce"),
        "clicks":         pd.to_numeric(df["metrics_clicks"], errors="coerce").fillna(0),
        "impressions":    pd.to_numeric(df["metrics_impressions"], errors="coerce").fillna(0),
        "conversions":    pd.to_numeric(df["metrics_conversions"], errors="coerce").fillna(0),
        "daily_budget":   pd.to_numeric(df["campaign_budget_amount"], errors="coerce").fillna(0),
    })
    return _add_flags(out)


def load_bing(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({
        "date":           pd.to_datetime(df["TimePeriod"], errors="coerce"),
        "channel":        "bing",
        "campaign_id":    df["CampaignId"].astype(str),
        "campaign_name":  df["CampaignName"],
        "campaign_type":  df["CampaignType"].map(
                              lambda x: _normalize_campaign_type(x, "bing")),
        "audience_segment": df["CampaignName"].map(_parse_audience_segment),
        "spend":          pd.to_numeric(df["Spend"], errors="coerce"),
        "revenue":        pd.to_numeric(df["Revenue"], errors="coerce"),
        "clicks":         pd.to_numeric(df["Clicks"], errors="coerce").fillna(0),
        "impressions":    pd.to_numeric(df["Impressions"], errors="coerce").fillna(0),
        "conversions":    pd.to_numeric(df["Conversions"], errors="coerce").fillna(0),
        "daily_budget":   pd.to_numeric(df["DailyBudget"], errors="coerce").fillna(0),
    })
    return _add_flags(out)


def load_meta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    revenue = (
        pd.to_numeric(df["conversion"], errors="coerce")
        if META_CONVERSION_AS_REVENUE
        else pd.Series(0.0, index=df.index)
    )
    out = pd.DataFrame({
        "date":           pd.to_datetime(df["date_start"], errors="coerce"),
        "channel":        "meta",
        "campaign_id":    df["campaign_id"].astype(str),
        "campaign_name":  df["campaign_name"],
        "campaign_type":  df["campaign_name"].map(_infer_meta_campaign_type),
        "audience_segment": df["campaign_name"].map(_parse_audience_segment),
        "spend":          pd.to_numeric(df["spend"], errors="coerce"),
        "revenue":        revenue,
        "clicks":         pd.to_numeric(df["clicks"], errors="coerce").fillna(0),
        "impressions":    pd.to_numeric(df["impressions"], errors="coerce").fillna(0),
        "conversions":    pd.to_numeric(df["conversion"], errors="coerce").fillna(0),
        "daily_budget":   pd.to_numeric(df["daily_budget"], errors="coerce").fillna(0),
    })
    return _add_flags(out)


# ─── Discovery + combined loader ─────────────────────────────────────────────

def discover_csvs(data_dir: Path) -> dict[str, Path]:
    """Auto-detect Google / Bing / Meta CSVs by name pattern."""
    files = {p.name.lower(): p for p in data_dir.glob("*.csv")}
    mapping: dict[str, Path] = {}
    for name, path in files.items():
        if "google" in name:
            mapping["google"] = path
        elif "bing" in name or "microsoft" in name or "ms_" in name:
            mapping["bing"] = path
        elif "meta" in name or "facebook" in name:
            mapping["meta"] = path
    return mapping


def load_all(data_dir: Path | str) -> pd.DataFrame:
    """Load, clean, deduplicate and concatenate all channel CSVs."""
    data_dir = Path(data_dir)
    csvs = discover_csvs(data_dir)
    frames: list[pd.DataFrame] = []
    if "google" in csvs:
        frames.append(load_google(csvs["google"]))
    if "bing" in csvs:
        frames.append(load_bing(csvs["bing"]))
    if "meta" in csvs:
        frames.append(load_meta(csvs["meta"]))
    if not frames:
        raise FileNotFoundError(f"No recognized CSV files in {data_dir}")

    combined = pd.concat(frames, ignore_index=True)

    # ── Global cleaning ──────────────────────────────────────────────────────
    # 1. Drop unparseable dates
    combined = combined.dropna(subset=["date"])

    # 2. Drop future dates
    today = pd.Timestamp.today().normalize()
    future_mask = combined["date"] > today
    n_future = int(future_mask.sum())
    if n_future > 0:
        import warnings
        warnings.warn(f"Dropping {n_future} rows with future dates.")
    combined = combined[~future_mask]

    # 3. Deduplicate on (channel, campaign_id, date) — keep first occurrence
    before = len(combined)
    combined = combined.drop_duplicates(subset=["channel", "campaign_id", "date"], keep="first")
    n_dupes = before - len(combined)
    if n_dupes > 0:
        import warnings
        warnings.warn(f"Dropped {n_dupes} duplicate (channel, campaign_id, date) rows.")

    return combined[CANONICAL_COLUMNS]
