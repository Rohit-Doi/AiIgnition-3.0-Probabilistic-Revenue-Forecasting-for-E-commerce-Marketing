"""Glue between raw CSVs and forecasts: ingest -> validate -> hierarchy -> features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.features import latest_features
from src.forecast import META_COLS
from src.hierarchy import Hierarchy, build_hierarchy
from src.load_data import load_all
from src.validate import ValidationReport, clean, validate


@dataclass
class Prepared:
    cleaned: pd.DataFrame          # canonical daily rows, one per (channel, campaign, date)
    report: ValidationReport
    hier: Hierarchy


def prepare_data(data_dir: Path | str) -> Prepared:
    """Read every channel export in `data_dir`, validate, clean and build the hierarchy."""
    raw = load_all(data_dir)
    report = validate(raw)
    cleaned = clean(raw, report)
    return Prepared(cleaned=cleaned, report=report, hier=build_hierarchy(cleaned))


def feature_table(hier: Hierarchy, horizons: list[int] | None = None) -> pd.DataFrame:
    """Model inputs at the forecast origin: one row per series and horizon, with its identity."""
    feats = latest_features(hier, horizons)
    return feats.merge(hier.meta[META_COLS], on="series_id", how="left")

