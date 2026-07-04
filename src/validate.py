"""Data validation with structured report."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime

import pandas as pd

from src.config import CAMPAIGN_TYPE_ALLOWLIST


@dataclass
class ValidationReport:
    total_rows: int = 0
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
    passed: bool = True
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


REQUIRED_COLUMNS = [
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
]


def validate(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport(total_rows=len(df))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.missing_columns = missing
    if missing:
        report.passed = False
        report.messages.append(f"Missing columns: {missing}")
        return report

    today = pd.Timestamp(datetime.utcnow().date())
    report.unparseable_dates = int(df["date"].isna().sum())
    report.future_dates = int((df["date"] > today).sum())
    report.negative_spend = int((df["spend"] < 0).sum())
    report.negative_revenue = int((df["revenue"] < 0).sum())
    report.null_spend = int(df["spend"].isna().sum())
    report.null_revenue = int(df["revenue"].isna().sum())
    report.duplicate_campaign_date = int(
        df.duplicated(subset=["campaign_id", "date", "channel"]).sum()
    )
    report.unknown_campaign_types = int(
        (~df["campaign_type"].isin(CAMPAIGN_TYPE_ALLOWLIST)).sum()
    )
    report.zero_spend_positive_revenue = int(
        ((df["spend"] == 0) & (df["revenue"] > 0)).sum()
    )

    if report.unparseable_dates:
        report.messages.append(f"{report.unparseable_dates} unparseable dates")
    if report.future_dates:
        report.messages.append(f"{report.future_dates} future dates flagged")
    if report.duplicate_campaign_date:
        report.messages.append(
            f"{report.duplicate_campaign_date} duplicate (channel, campaign_id, date) rows"
        )
    return report


def clean(df: pd.DataFrame, report: ValidationReport | None = None) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["date"])
    out = out[out["date"] <= pd.Timestamp(datetime.utcnow().date())]

    neg_spend = (out["spend"] < 0).sum()
    neg_rev = (out["revenue"] < 0).sum()
    out["spend"] = out["spend"].clip(lower=0)
    out["revenue"] = out["revenue"].clip(lower=0)

    if report:
        if neg_spend:
            report.messages.append(f"Clipped {neg_spend} negative spend rows to 0")
        if neg_rev:
            report.messages.append(f"Clipped {neg_rev} negative revenue rows to 0")

    budget_mean = out.groupby("campaign_id")["daily_budget"].transform("mean")
    out["daily_budget"] = out["daily_budget"].fillna(budget_mean).fillna(0)

    out = out.drop_duplicates(subset=["channel", "campaign_id", "date"], keep="first")
    return out.reset_index(drop=True)
