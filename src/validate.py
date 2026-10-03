"""Data validation and cleaning, with a report an analyst can act on.

Besides row-level checks (dates, negatives, duplicates) the report covers
*campaign consistency* - the things that silently distort a forecast when
platform exports are stitched together: one campaign id under several names or
types, channels whose exports end on different days, days missing from the
middle of a channel's history, and spend that produced no tracked revenue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from src.config import CAMPAIGN_TYPE_ALLOWLIST, MATURITY_LAG_DAYS

REQUIRED_COLUMNS = [
    "date", "channel", "campaign_id", "campaign_name", "campaign_type", "audience_segment",
    "spend", "revenue", "clicks", "impressions", "conversions", "daily_budget",
]


@dataclass
class ValidationReport:
    total_rows: int = 0
    date_min: str = ""
    date_max: str = ""
    channels: list[str] = field(default_factory=list)
    n_campaigns: int = 0
    missing_columns: list[str] = field(default_factory=list)
    unparseable_dates: int = 0
    future_dates: int = 0
    negative_spend: int = 0
    negative_revenue: int = 0
    null_spend: int = 0
    null_revenue: int = 0
    duplicate_campaign_date: int = 0
    unknown_campaign_types: int = 0
    zero_spend_positive_revenue: int = 0
    spend_without_revenue_campaigns: int = 0
    campaigns_with_multiple_names: int = 0
    campaigns_with_multiple_types: int = 0
    names_shared_by_ids: int = 0
    channel_coverage: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)      # {check, status: ok|warn|fail, detail}
    passed: bool = True
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def add(self, check: str, status: str, detail: str) -> None:
        self.checks.append({"check": check, "status": status, "detail": detail})
        if status != "ok":
            self.messages.append(detail)
        if status == "fail":
            self.passed = False


def validate(df: pd.DataFrame, today: pd.Timestamp | None = None) -> ValidationReport:
    report = ValidationReport(total_rows=len(df))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.missing_columns = missing
    if missing:
        report.add("Schema", "fail", f"Missing columns: {missing}")
        return report
    report.add("Schema", "ok", "All canonical columns present after harmonisation.")
    if df.empty:
        report.add("Rows", "fail", "No rows with a parseable date.")
        return report

    stats = df.attrs.get("load_stats", {})
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    report.date_min = str(df["date"].min().date())
    report.date_max = str(df["date"].max().date())
    report.channels = sorted(df["channel"].unique().tolist())
    report.n_campaigns = int(df.groupby(["channel", "campaign_id"]).ngroups)

    report.unparseable_dates = int(stats.get("unparseable_dates", df["date"].isna().sum()))
    report.future_dates = int((df["date"] > today).sum())
    report.negative_spend = int((df["spend"] < 0).sum())
    report.negative_revenue = int((df["revenue"] < 0).sum())
    report.null_spend = int(df["spend"].isna().sum())
    report.null_revenue = int(df["revenue"].isna().sum())
    report.duplicate_campaign_date = int(stats.get(
        "duplicates_dropped", df.duplicated(subset=["channel", "campaign_id", "date"]).sum()))
    report.unknown_campaign_types = int(
        (~df["campaign_type"].isin(CAMPAIGN_TYPE_ALLOWLIST - {"UNKNOWN"})).sum())
    report.zero_spend_positive_revenue = int(((df["spend"] == 0) & (df["revenue"] > 0)).sum())

    def count_check(name: str, n: int, what: str, bad: str = "warn") -> None:
        report.add(name, "ok" if n == 0 else bad, f"{n:,} {what}" if n else f"No {what}.")

    count_check("Dates", report.unparseable_dates, "rows with an unparseable date (dropped)")
    count_check("Future dates", report.future_dates, "rows dated in the future (dropped)")
    count_check("Duplicates", report.duplicate_campaign_date,
                "duplicate (channel, campaign, date) rows (first kept)")
    count_check("Negative values", report.negative_spend + report.negative_revenue,
                "rows with negative spend or revenue (clipped to 0)")
    count_check("Campaign types", report.unknown_campaign_types,
                "rows with an unrecognised campaign type (grouped as UNKNOWN)")

    # ── campaign consistency ────────────────────────────────────────────────
    per_id = df.groupby(["channel", "campaign_id"]).agg(
        names=("campaign_name", "nunique"), types=("campaign_type", "nunique"),
        spend=("spend", "sum"), revenue=("revenue", "sum"))
    report.campaigns_with_multiple_names = int((per_id["names"] > 1).sum())
    report.campaigns_with_multiple_types = int((per_id["types"] > 1).sum())
    report.names_shared_by_ids = int(
        (df.groupby(["channel", "campaign_name"])["campaign_id"].nunique() > 1).sum())
    report.spend_without_revenue_campaigns = int(((per_id["spend"] >= 100) & (per_id["revenue"] == 0)).sum())

    count_check("Campaign naming", report.campaigns_with_multiple_names,
                "campaign ids appear under more than one name")
    count_check("Campaign typing", report.campaigns_with_multiple_types,
                "campaign ids change campaign type over time (latest type is used)")
    count_check("Shared names", report.names_shared_by_ids,
                "campaign names are shared by several ids (merged into one forecast row)")
    count_check("Untracked spend", report.spend_without_revenue_campaigns,
                "campaigns spent $100+ without any attributed revenue")
    count_check("Revenue without spend", report.zero_spend_positive_revenue,
                "rows carry revenue on a zero-spend day (kept; typical of delayed attribution)")

    # ── channel coverage ────────────────────────────────────────────────────
    newest = df["date"].max()
    for channel, g in df.groupby("channel"):
        days = pd.DatetimeIndex(g["date"].unique())
        span = (days.max() - days.min()).days + 1
        report.channel_coverage.append({
            "channel": channel,
            "first_date": str(days.min().date()),
            "last_date": str(days.max().date()),
            "days_with_data": int(len(days)),
            "missing_days": int(span - len(days)),
            "days_behind_newest": int((newest - days.max()).days),
            "campaigns": int(g["campaign_id"].nunique()),
            "spend": float(g["spend"].sum()),
            "revenue": float(g["revenue"].sum()),
        })
    behind = [c for c in report.channel_coverage if 0 < c["days_behind_newest"]]
    report.add("Channel alignment", "warn" if behind else "ok",
               ("Exports end on different days: "
                + ", ".join(f"{c['channel']} ends {c['last_date']}" for c in report.channel_coverage)
                + ". The forecast origin is the last day every reporting channel covers.")
               if behind else "All channels end on the same day.")
    gaps = [c for c in report.channel_coverage if c["missing_days"] > 0]
    report.add("Calendar gaps", "warn" if gaps else "ok",
               ("Days with no rows inside a channel's history (treated as zero activity): "
                + ", ".join(f"{c['channel']} {c['missing_days']}" for c in gaps))
               if gaps else "No missing days inside any channel's history.")
    report.add("Conversion lag", "ok",
               f"Revenue for the final {MATURITY_LAG_DAYS} days is nowcast from spend x trailing ROAS "
               "because platforms keep attributing conversions after the click.")
    return report


def clean(df: pd.DataFrame, report: ValidationReport | None = None,
          today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Apply the fixes the report describes; returns one row per (channel, campaign, date)."""
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    out = df.dropna(subset=["date"]).copy()
    out = out[out["date"] <= today]
    out["spend"] = out["spend"].clip(lower=0).fillna(0)
    out["revenue"] = out["revenue"].clip(lower=0).fillna(0)
    for col in ("clicks", "impressions", "conversions"):
        out[col] = out[col].fillna(0)

    out = out.sort_values(["channel", "campaign_id", "date"])
    budget = out.groupby(["channel", "campaign_id"])["daily_budget"]
    out["daily_budget"] = budget.ffill().fillna(budget.transform("mean")).fillna(0)
    out["campaign_type"] = out["campaign_type"].where(
        out["campaign_type"].isin(CAMPAIGN_TYPE_ALLOWLIST), "UNKNOWN")

    out = out.drop_duplicates(subset=["channel", "campaign_id", "date"], keep="first")
    return out.reset_index(drop=True)
