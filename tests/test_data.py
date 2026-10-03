"""Ingestion, validation and hierarchy construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_exports
from src.hierarchy import build_hierarchy, resolve_origin
from src.load_data import CANONICAL_COLUMNS, discover_csvs, load_all
from src.validate import clean, validate


def test_schema_is_harmonised(exports):
    df = load_all(exports)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert set(df["channel"]) == {"google", "meta", "bing"}
    assert (df["spend"] >= 0).all() and (df["revenue"] >= 0).all()
    # Google cost arrives in micros
    assert df.loc[df["channel"] == "google", "spend"].max() < 5_000


def test_files_are_recognised_by_header_not_name(exports, tmp_path):
    for i, f in enumerate(sorted(exports.glob("*.csv"))):
        (tmp_path / f"export_{i}.csv").write_bytes(f.read_bytes())
    assert set(discover_csvs(tmp_path)) == {"google", "meta", "bing"}
    assert len(load_all(tmp_path)) == len(load_all(exports))


def test_missing_channel_is_tolerated(exports, tmp_path):
    (tmp_path / "google.csv").write_bytes((exports / "google_ads_campaign_stats.csv").read_bytes())
    df = load_all(tmp_path)
    assert set(df["channel"]) == {"google"}


def test_no_csv_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all(tmp_path)


def test_dirty_rows_are_cleaned_and_reported(exports, tmp_path):
    g = pd.read_csv(exports / "google_ads_campaign_stats.csv", index_col=0)
    g = pd.concat([g, g.tail(25)])                                       # duplicates
    g.iloc[0, g.columns.get_loc("segments_date")] = "not-a-date"         # bad date
    g.iloc[1, g.columns.get_loc("metrics_conversions_value")] = -50.0    # negative revenue
    g.iloc[2:6, g.columns.get_loc("campaign_advertising_channel_type")] = "LOCAL_SERVICES"
    g.to_csv(tmp_path / "google_ads_campaign_stats.csv")

    raw = load_all(tmp_path)
    report = validate(raw)
    cleaned = clean(raw, report)
    assert report.duplicate_campaign_date == 25
    assert report.unparseable_dates == 1
    assert report.unknown_campaign_types == 4
    assert not cleaned.duplicated(["channel", "campaign_id", "date"]).any()
    assert (cleaned["revenue"] >= 0).all()
    assert any(c["status"] == "warn" for c in report.checks)


def test_meta_campaign_types_come_from_names(exports):
    meta = load_all(exports).query("channel == 'meta'")
    assert set(meta["campaign_type"]) == {"BRAND", "REMARKETING"}


def test_hierarchy_levels_add_up(prepared):
    hier = prepared.hier
    meta = hier.meta
    blended = hier.revenue[0]
    channels = hier.revenue[meta.index[meta["level"] == "channel"]].sum(axis=0)
    types = hier.revenue[meta.index[meta["level"] == "campaign_type"]].sum(axis=0)
    campaigns = hier.revenue[meta.index[meta["level"] == "campaign"]].sum(axis=0)
    for level in (channels, types, campaigns):
        np.testing.assert_allclose(level, blended, rtol=1e-9)
    assert (meta.loc[meta["level"] == "blended", "channel"] == "all").all()


def test_origin_is_last_day_every_channel_covers(tmp_path):
    make_exports(tmp_path, days=60)
    b = pd.read_csv(tmp_path / "bing_campaign_stats.csv", index_col=0)
    b[b["TimePeriod"] <= "2026-03-29"].to_csv(tmp_path / "bing_campaign_stats.csv")   # bing ends 2 days early
    df = clean(load_all(tmp_path))
    assert resolve_origin(df) == pd.Timestamp("2026-03-29")
    assert build_hierarchy(df).origin == pd.Timestamp("2026-03-29")
