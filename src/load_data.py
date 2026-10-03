"""Load and harmonise Google Ads / Microsoft Ads / Meta Ads exports into one schema.

Files are recognised by name first and by their header second, so renamed or
split exports still load.  Every platform is mapped to the canonical daily frame

    date, channel, campaign_id, campaign_name, campaign_type, audience_segment,
    spend, revenue, clicks, impressions, conversions, daily_budget

plus two anomaly flags that travel with the data into the validation report.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from src.config import META_CONVERSION_AS_REVENUE

CANONICAL_COLUMNS = [
    "date", "channel", "campaign_id", "campaign_name", "campaign_type", "audience_segment",
    "spend", "revenue", "clicks", "impressions", "conversions", "daily_budget",
    # anomaly flags (reported, never used as model features)
    "flag_zero_spend_nonzero_revenue", "flag_zero_revenue_nonzero_spend",
]

# A column that only that platform's export has.
_SIGNATURE = {"google": "segments_date", "bing": "TimePeriod", "meta": "date_start"}
_NAME_TOKENS = {
    "google": ("google",),
    "bing": ("bing", "microsoft", "ms_ads", "msads", "ms_"),
    "meta": ("meta", "facebook", "fb_"),
}


# ─── Parsing helpers ─────────────────────────────────────────────────────────

def _parse_audience_segment(name: str) -> str:
    """Audience segment from the campaign naming convention (NTM is tested before TM)."""
    if not isinstance(name, str):
        return "generic"
    lower = name.lower()
    if any(tok in lower for tok in ("_ntm_", "ntm_", " ntm ", "nonbrand", "non_brand", "non-brand")):
        return "nonbrand"
    if any(tok in lower for tok in ("_tm_", " tm ", "tm_", "brand")):
        return "brand"
    if "prospecting" in lower or "prosp" in lower:
        return "prospecting"
    if "remarketing" in lower or "retarg" in lower:
        return "remarketing"
    if "dpa" in lower or "dynamic" in lower:
        return "dpa"
    return "generic"


_TYPE_MAP = {
    "SEARCH": "SEARCH",
    "PERFORMANCE_MAX": "PERFORMANCE_MAX", "PERFORMANCEMAX": "PERFORMANCE_MAX", "PMAX": "PERFORMANCE_MAX",
    "DISPLAY": "DISPLAY",
    "VIDEO": "VIDEO",
    "DEMAND_GEN": "DEMAND_GEN", "DEMANDGEN": "DEMAND_GEN",
    "SHOPPING": "SHOPPING",
    "AUDIENCE": "DISPLAY",          # Microsoft Audience Network
    "REMARKETING": "REMARKETING",
    "PROSPECTING": "PROSPECTING",
    "BRAND": "BRAND",
}


def _normalize_campaign_type(raw: object) -> str:
    if not isinstance(raw, str):
        return "UNKNOWN"
    return _TYPE_MAP.get(raw.strip().upper().replace(" ", "_").replace("-", "_"), "UNKNOWN")


def _infer_meta_campaign_type(name: object) -> str:
    """Meta exports carry no campaign type; infer it from the naming convention."""
    if not isinstance(name, str):
        return "DISPLAY"
    lower = name.lower()
    if "video" in lower:
        return "VIDEO"
    if any(tok in lower for tok in ("brand", "_tm_", " tm ", "tm_")):
        return "BRAND"
    if "remarketing" in lower or "retarg" in lower or "dpa" in lower:
        return "REMARKETING"
    if "prospecting" in lower or "prosp" in lower:
        return "PROSPECTING"
    return "DISPLAY"


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column, or NaN when the export does not have it."""
    if col not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _dates(df: pd.DataFrame, col: str) -> pd.Series:
    """Parse a date column: ISO first (what the platforms export), then free-form as a fallback."""
    if col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    raw = df[col].astype(str).str.strip()
    parsed = pd.to_datetime(raw, format="ISO8601", errors="coerce")
    if parsed.isna().mean() > 0.5:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(raw, errors="coerce")
    return parsed


def _text(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return df[col]


def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Clip spend / revenue at zero and flag rows where only one of them is non-zero."""
    df = df.copy()
    df["spend"] = df["spend"].clip(lower=0).fillna(0)
    df["revenue"] = df["revenue"].clip(lower=0).fillna(0)
    df["flag_zero_spend_nonzero_revenue"] = (df["spend"] == 0) & (df["revenue"] > 0)
    df["flag_zero_revenue_nonzero_spend"] = (df["revenue"] == 0) & (df["spend"] > 0)
    return df


# ─── Per-platform loaders ────────────────────────────────────────────────────

def load_google(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    name = _text(df, "campaign_name")
    return _add_flags(pd.DataFrame({
        "date": _dates(df, "segments_date"),
        "channel": "google",
        "campaign_id": _text(df, "campaign_id").astype(str),
        "campaign_name": name,
        "campaign_type": _text(df, "campaign_advertising_channel_type").map(_normalize_campaign_type),
        "audience_segment": name.map(_parse_audience_segment),
        "spend": _num(df, "metrics_cost_micros") / 1_000_000,
        "revenue": _num(df, "metrics_conversions_value"),
        "clicks": _num(df, "metrics_clicks").fillna(0),
        "impressions": _num(df, "metrics_impressions").fillna(0),
        "conversions": _num(df, "metrics_conversions").fillna(0),
        "daily_budget": _num(df, "campaign_budget_amount"),
    }))


def load_bing(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    name = _text(df, "CampaignName")
    return _add_flags(pd.DataFrame({
        "date": _dates(df, "TimePeriod"),
        "channel": "bing",
        "campaign_id": _text(df, "CampaignId").astype(str),
        "campaign_name": name,
        "campaign_type": _text(df, "CampaignType").map(_normalize_campaign_type),
        "audience_segment": name.map(_parse_audience_segment),
        "spend": _num(df, "Spend"),
        "revenue": _num(df, "Revenue"),
        "clicks": _num(df, "Clicks").fillna(0),
        "impressions": _num(df, "Impressions").fillna(0),
        "conversions": _num(df, "Conversions").fillna(0),
        "daily_budget": _num(df, "DailyBudget"),
    }))


def load_meta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    name = _text(df, "campaign_name")
    value = _num(df, "conversion")
    return _add_flags(pd.DataFrame({
        "date": _dates(df, "date_start"),
        "channel": "meta",
        "campaign_id": _text(df, "campaign_id").astype(str),
        "campaign_name": name,
        "campaign_type": name.map(_infer_meta_campaign_type),
        "audience_segment": name.map(_parse_audience_segment),
        "spend": _num(df, "spend"),
        "revenue": value if META_CONVERSION_AS_REVENUE else 0.0,
        "clicks": _num(df, "clicks").fillna(0),
        "impressions": _num(df, "impressions").fillna(0),
        "conversions": value.fillna(0),
        "daily_budget": _num(df, "daily_budget"),
    }))


_LOADERS = {"google": load_google, "bing": load_bing, "meta": load_meta}


# ─── Discovery + combined loader ─────────────────────────────────────────────

def _channel_of(path: Path) -> str | None:
    """Platform a CSV belongs to: by its header first, then by its file name."""
    try:
        header = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        header = set()
    for channel, column in _SIGNATURE.items():
        if column in header:
            return channel
    lower = path.name.lower()
    for channel, tokens in _NAME_TOKENS.items():
        if any(tok in lower for tok in tokens):
            return channel
    return None


def discover_csvs(data_dir: Path) -> dict[str, list[Path]]:
    """{channel: [files]} for every recognised CSV in the folder (searched recursively)."""
    found: dict[str, list[Path]] = {}
    for path in sorted(Path(data_dir).rglob("*.csv")):
        channel = _channel_of(path)
        if channel is None:
            warnings.warn(f"Skipping unrecognised file: {path.name}")
            continue
        found.setdefault(channel, []).append(path)
    return found


def load_all(data_dir: Path | str) -> pd.DataFrame:
    """Load, clean, de-duplicate and concatenate every recognised channel CSV."""
    data_dir = Path(data_dir)
    csvs = discover_csvs(data_dir)
    frames = [_LOADERS[channel](path) for channel, paths in csvs.items() for path in paths]
    if not frames:
        raise FileNotFoundError(f"No Google / Microsoft / Meta campaign CSVs found in {data_dir}")

    combined = pd.concat(frames, ignore_index=True)
    n_raw = len(combined)
    combined = combined.dropna(subset=["date"])
    n_bad_dates = n_raw - len(combined)
    combined["date"] = combined["date"].dt.normalize()

    before = len(combined)
    combined = combined.drop_duplicates(subset=["channel", "campaign_id", "date"], keep="first")
    n_dupes = before - len(combined)

    combined = combined[CANONICAL_COLUMNS].reset_index(drop=True)
    combined.attrs["load_stats"] = {
        "files": {ch: [p.name for p in paths] for ch, paths in csvs.items()},
        "rows_read": int(n_raw),
        "unparseable_dates": int(n_bad_dates),
        "duplicates_dropped": int(n_dupes),
    }
    return combined
