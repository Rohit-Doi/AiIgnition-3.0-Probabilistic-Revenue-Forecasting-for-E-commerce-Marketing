"""Shared fixtures: a small synthetic account in the three platform export schemas."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["AIGNITION_DISABLE_LLM"] = "1"       # tests never call an LLM

MODEL_PATH = ROOT / "pickle" / "model.pkl"


def make_exports(folder: Path, days: int = 150, seed: int = 0, end: str = "2026-03-31") -> Path:
    """Write Google / Microsoft / Meta CSVs with known structure (ROAS about 4, 6 and 3)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=days, freq="D")
    folder.mkdir(parents=True, exist_ok=True)

    def series(base_spend: float, roas: float) -> tuple[np.ndarray, np.ndarray]:
        spend = base_spend * (1 + 0.15 * np.sin(np.arange(days) / 7)) * rng.lognormal(0, 0.15, days)
        return spend, spend * roas * rng.lognormal(0, 0.25, days)

    google = []
    for cid, (name, ctype, spend, roas) in enumerate([
            ("Search_TM_Campaign_01", "SEARCH", 300, 5.0), ("Search_NTM_Campaign_02", "SEARCH", 120, 2.5),
            ("Pmax_NTM_Campaign_01", "PERFORMANCE_MAX", 400, 4.0), ("Shopping_NTM_Campaign_01", "SHOPPING", 200, 6.0)]):
        sp, rev = series(spend, roas)
        google.append(pd.DataFrame({
            "campaign_id": 1000 + cid, "segments_date": dates.strftime("%Y-%m-%d"),
            "metrics_clicks": (sp / 1.5).astype(int), "metrics_conversions": rev / 80,
            "metrics_cost_micros": (sp * 1_000_000).astype("int64"), "metrics_impressions": (sp * 20).astype(int),
            "metrics_video_views": 0, "metrics_conversions_value": rev,
            "campaign_advertising_channel_type": ctype, "campaign_budget_amount": spend * 1.2,
            "campaign_name": name}))
    pd.concat(google).to_csv(folder / "google_ads_campaign_stats.csv")

    meta = []
    for cid, (name, spend, roas) in enumerate([("Prospecting_Brand_Campaign_01", 90, 6.0),
                                               ("Remarketing_DPA_Campaign_01", 60, 7.0)]):
        sp, rev = series(spend, roas)
        meta.append(pd.DataFrame({
            "campaign_id": 2000 + cid, "date_start": dates.strftime("%Y-%m-%d"), "cpc": 1.0, "cpm": 10.0,
            "ctr": 1.0, "reach": 0.0, "spend": sp, "clicks": sp / 1.2, "impressions": sp * 30, "conversion": rev,
            "daily_budget": np.nan, "campaign_name": name}))
    pd.concat(meta).to_csv(folder / "meta_ads_campaign_stats.csv")

    sp, rev = series(40, 3.0)
    pd.DataFrame({
        "CampaignId": 3000, "TimePeriod": dates.strftime("%Y-%m-%d"), "Revenue": rev, "Spend": sp,
        "Clicks": (sp / 1.1).astype(int), "Impressions": (sp * 15).astype(int), "Conversions": rev / 70,
        "CampaignType": "Search", "DailyBudget": 50.0, "CampaignName": "Search_TM_Campaign_02",
    }).to_csv(folder / "bing_campaign_stats.csv")
    return folder


@pytest.fixture(scope="session")
def exports(tmp_path_factory) -> Path:
    return make_exports(tmp_path_factory.mktemp("exports"))


@pytest.fixture(scope="session")
def bundle() -> dict:
    from src.model import load_bundle

    if not MODEL_PATH.exists():
        pytest.skip("pickle/model.pkl not found; run `python -m src.train` first")
    return load_bundle(MODEL_PATH)


@pytest.fixture(scope="session")
def prepared(exports):
    from src.pipeline import prepare_data

    return prepare_data(exports)


@pytest.fixture(scope="session")
def baseline(prepared, bundle) -> pd.DataFrame:
    from src.forecast import forecast
    from src.pipeline import feature_table

    return forecast(bundle, feature_table(prepared.hier))
