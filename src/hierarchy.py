"""Daily forecasting hierarchy: blended -> channel -> campaign type -> campaign.

Every node of the hierarchy is a *series* with dense daily revenue / spend arrays
over one shared calendar.  All downstream feature engineering works on these
arrays with cumulative sums, so features for every series at a forecast origin
are computed in one vectorised pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVELS = ["blended", "channel", "campaign_type", "campaign"]
LEVEL_CODE = {name: i for i, name in enumerate(LEVELS)}

# A channel whose last row is older than this (vs. the newest row in the extract)
# is treated as inactive rather than as "not yet reported".
_STALE_CHANNEL_DAYS = 14


@dataclass
class Hierarchy:
    """Dense daily matrices for every node of the forecasting hierarchy."""

    dates: pd.DatetimeIndex          # shared daily calendar (length D)
    meta: pd.DataFrame               # one row per series (see build_hierarchy)
    revenue: np.ndarray              # (n_series, D)
    spend: np.ndarray                # (n_series, D)
    clicks: np.ndarray               # (n_series, D)

    @property
    def n_series(self) -> int:
        return len(self.meta)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def origin(self) -> pd.Timestamp:
        """Last fully observed day — forecasts cover the days after it."""
        return self.dates[-1]

    def index_of(self, date: pd.Timestamp | str) -> int:
        return int(self.dates.get_loc(pd.Timestamp(date)))

    def truncate(self, end: pd.Timestamp | str) -> "Hierarchy":
        """Copy of the hierarchy as it would have looked at `end` (inclusive)."""
        stop = self.index_of(end) + 1
        return Hierarchy(
            dates=self.dates[:stop],
            meta=self.meta.copy(),
            revenue=self.revenue[:, :stop],
            spend=self.spend[:, :stop],
            clicks=self.clicks[:, :stop],
        )


def resolve_origin(df: pd.DataFrame) -> pd.Timestamp:
    """Last day on which every still-reporting channel has data.

    Platform exports rarely end on the same day, and the final day of an extract
    is usually partial.  Using the earliest "last day" among channels that are
    still reporting keeps a half-loaded day out of every run-rate.
    """
    last_by_channel = df.groupby("channel")["date"].max()
    newest = last_by_channel.max()
    live = last_by_channel[last_by_channel >= newest - pd.Timedelta(days=_STALE_CHANNEL_DAYS)]
    return pd.Timestamp(live.min())


def build_hierarchy(df: pd.DataFrame, origin: pd.Timestamp | None = None) -> Hierarchy:
    """Build the four-level daily hierarchy from the canonical daily frame.

    Campaign series are keyed by (channel, campaign_name) because the submission
    format identifies campaigns by name; ids that share a name are merged.  A
    campaign's type is the one it reported most recently.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    origin = pd.Timestamp(origin) if origin is not None else resolve_origin(df)
    df = df[df["date"] <= origin]
    if df.empty:
        raise ValueError("No rows on or before the forecast origin.")

    df["campaign_name"] = df["campaign_name"].fillna("").astype(str)
    blank = df["campaign_name"].str.strip() == ""
    df.loc[blank, "campaign_name"] = "campaign_" + df.loc[blank, "campaign_id"].astype(str)

    latest_type = (
        df.sort_values("date")
        .groupby(["channel", "campaign_name"])["campaign_type"]
        .last()
        .rename("latest_type")
        .reset_index()
    )
    df = df.merge(latest_type, on=["channel", "campaign_name"], how="left")
    df["campaign_type"] = df["latest_type"]

    dates = pd.date_range(df["date"].min(), origin, freq="D")
    day_idx = dates.get_indexer(df["date"])

    campaigns = (
        df.groupby(["channel", "campaign_type", "campaign_name"], sort=True)
        .size()
        .reset_index()[["channel", "campaign_type", "campaign_name"]]
    )
    campaigns["cid"] = np.arange(len(campaigns))
    df = df.merge(campaigns, on=["channel", "campaign_type", "campaign_name"], how="left")

    n_c, n_d = len(campaigns), len(dates)
    measures = {}
    for col in ("revenue", "spend", "clicks"):
        mat = np.zeros((n_c, n_d))
        np.add.at(mat, (df["cid"].to_numpy(), day_idx), df[col].fillna(0.0).to_numpy(dtype=float))
        measures[col] = mat

    rows: list[dict] = [dict(level="blended", channel="all", campaign_type="", campaign_name="")]
    members: list[np.ndarray] = [np.arange(n_c)]

    for ch in sorted(campaigns["channel"].unique()):
        rows.append(dict(level="channel", channel=ch, campaign_type="", campaign_name=""))
        members.append(campaigns.index[campaigns["channel"] == ch].to_numpy())

    for (ch, ct), grp in campaigns.groupby(["channel", "campaign_type"], sort=True):
        rows.append(dict(level="campaign_type", channel=ch, campaign_type=ct, campaign_name=""))
        members.append(grp.index.to_numpy())

    for i, c in campaigns.iterrows():
        rows.append(dict(level="campaign", channel=c["channel"], campaign_type=c["campaign_type"],
                         campaign_name=c["campaign_name"]))
        members.append(np.array([i]))

    meta = pd.DataFrame(rows)
    meta["series_id"] = np.arange(len(meta))
    meta["level_code"] = meta["level"].map(LEVEL_CODE).astype(int)
    meta["key"] = meta["level"] + "|" + meta["channel"] + "|" + meta["campaign_type"] + "|" + meta["campaign_name"]

    key_to_id = dict(zip(meta["key"], meta["series_id"]))
    parents = []
    for _, r in meta.iterrows():
        if r["level"] == "blended":
            parents.append(-1)
        elif r["level"] == "channel":
            parents.append(0)
        elif r["level"] == "campaign_type":
            parents.append(key_to_id[f"channel|{r['channel']}||"])
        else:
            parents.append(key_to_id[f"campaign_type|{r['channel']}|{r['campaign_type']}|"])
    meta["parent_id"] = parents
    meta["channel_id"] = [
        key_to_id.get(f"channel|{ch}||", 0) if lvl != "blended" else 0
        for ch, lvl in zip(meta["channel"], meta["level"])
    ]

    stacked = {name: np.vstack([mat[m].sum(axis=0) for m in members]) for name, mat in measures.items()}
    return Hierarchy(dates=dates, meta=meta.reset_index(drop=True), **stacked)
