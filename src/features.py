"""Causal origin-level features for direct multi-horizon forecasting.

A *sample* is (series, forecast origin t, horizon h).  Every feature is computed
from data observed on or before t; the target is the series total over the h days
after t.  The model never predicts raw dollars: it predicts the log-ratio between
the future total and a run-rate baseline, which makes one global model usable
across series that differ in size by four orders of magnitude.

Seasonality enters as *analog ratios*: what happened to the same window in other
years, relative to the run-rate just before it.  For training samples the analog
year may lie after the sample (leave-one-year-out) as long as it lies before the
training cutoff, so no fold ever sees data beyond its own cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    HORIZONS,
    MATURITY_LAG_DAYS,
    NOWCAST_ROAS_CAP,
    PEAK_SEASON,
    SEASON_OFFSET_DAYS,
    SEASON_SHIFT_DAYS,
    SEASON_YEARS,
)
from src.hierarchy import Hierarchy

RUN_WINDOWS = [7, 14, 28, 56, 91, 182]
BASE_WINDOW = 28
MIN_HISTORY_DAYS = 1

CHANNEL_CODES = {"all": 0, "google": 1, "meta": 2, "bing": 3}
TYPE_CODES = {
    "": 0, "SEARCH": 1, "PERFORMANCE_MAX": 2, "SHOPPING": 3, "DISPLAY": 4, "VIDEO": 5,
    "DEMAND_GEN": 6, "BRAND": 7, "REMARKETING": 8, "PROSPECTING": 9, "UNKNOWN": 10,
}

FEATURE_COLS = [
    # horizon + identity
    "horizon", "level_code", "channel_code", "type_code",
    # size
    "log_rev_28", "log_sp_28", "log_share_parent",
    # revenue momentum (log ratios of run-rates vs the 28-day base)
    "r7_28", "r14_28", "r56_28", "r91_28", "r182_28",
    # spend momentum
    "s7_28", "s14_28", "s56_28", "s91_28", "s182_28",
    # efficiency
    "log_roas_28", "log_roas_91", "log_roas_7",
    # stability / intermittency
    "rev_cv_8w", "sp_cv_8w", "zero_rev_share_28", "zero_sp_share_28",
    "log_days_since_spend", "log_age_days",
    # context: parent + portfolio momentum
    "parent_r7_28", "parent_r91_28", "parent_s7_28", "port_r7_28", "port_r91_28", "port_s7_28",
    # seasonality analogs (log ratios)
    "seas_port_rev", "seas_port_sp", "seas_chan_rev", "seas_chan_sp", "seas_own_rev", "seas_own_sp",
    "seas_n_years",
    # calendar of the target window
    "peak_share_target", "peak_share_base", "target_doy_sin", "target_doy_cos",
]

CATEGORICAL_COLS = ["channel_code", "type_code"]


def _cumsum(mat: np.ndarray) -> np.ndarray:
    """Row-wise cumulative sum with a leading zero column: cs[:, i] = sum of first i days."""
    out = np.zeros((mat.shape[0], mat.shape[1] + 1))
    np.cumsum(mat, axis=1, out=out[:, 1:])
    return out


def _wsum(cs: np.ndarray, a: int, b: int) -> np.ndarray:
    """Sum over day indices a..b inclusive (empty window -> zeros)."""
    if b < a:
        return np.zeros(cs.shape[0])
    return cs[:, b + 1] - cs[:, a]


def _peak_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    (m0, d0), (m1, d1) = PEAK_SEASON
    md = dates.month * 100 + dates.day
    return np.asarray((md >= m0 * 100 + d0) & (md <= m1 * 100 + d1))


def _analog_ratio(cs: np.ndarray, t: int, h: int, shift: int, last_idx: int,
                  c_daily: np.ndarray | None) -> np.ndarray | None:
    """log(target-window total / base-window run-rate) in the year `shift` days away.

    Returns None when either window falls outside the observed calendar.  A series
    with no activity in both windows carries no seasonal information and gets NaN.
    """
    e = t - shift
    a0, b0 = e - BASE_WINDOW + 1, e
    a1, b1 = t + 1 - shift, t + h - shift
    if min(a0, a1) < 0 or max(b0, b1) > last_idx:
        return None
    den = _wsum(cs, a0, b0) / BASE_WINDOW * h
    num = _wsum(cs, a1, b1)
    # a reference account can differ in scale from the live one: smooth with its own level
    c = c_daily * h if c_daily is not None else 0.02 * (den + num) / 2 + 1e-9
    val = np.log((num + c) / (den + c))
    return np.where((den <= 0) & (num <= 0), np.nan, val)


def _analog_shifts() -> list[int]:
    """Day shifts to every analog window: each analog year, at each weekly offset around it."""
    return [SEASON_SHIFT_DAYS * k + off for k in SEASON_YEARS for off in SEASON_OFFSET_DAYS]


def _mean_over_years(values: list[np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        return np.full(n, np.nan), np.zeros(n)
    stack = np.stack(values)
    count = np.isfinite(stack).sum(axis=0)
    total = np.nansum(stack, axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan), count.astype(float)


# ─── Seasonal reference kept inside the model bundle ─────────────────────────

@dataclass
class SeasonalReference:
    """Daily blended + channel history stored with the trained model.

    Lets the pipeline compute seasonal analog ratios even when the data handed to
    it at run time is too short to contain the same window one year earlier.
    """

    dates: pd.DatetimeIndex
    keys: list[str]                  # "all" + channel names
    revenue: np.ndarray              # (len(keys), D)
    spend: np.ndarray

    @classmethod
    def from_hierarchy(cls, hier: Hierarchy) -> "SeasonalReference":
        top = hier.meta[hier.meta["level"].isin(["blended", "channel"])]
        ids = top["series_id"].to_numpy()
        return cls(dates=hier.dates, keys=top["channel"].tolist(),
                   revenue=hier.revenue[ids].copy(), spend=hier.spend[ids].copy())

    def to_dict(self) -> dict:
        return {"dates": [d.strftime("%Y-%m-%d") for d in self.dates], "keys": list(self.keys),
                "revenue": np.round(self.revenue, 2).tolist(), "spend": np.round(self.spend, 2).tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "SeasonalReference":
        return cls(dates=pd.DatetimeIndex(pd.to_datetime(d["dates"])), keys=list(d["keys"]),
                   revenue=np.asarray(d["revenue"], dtype=float), spend=np.asarray(d["spend"], dtype=float))

    def analog(self, origin: pd.Timestamp, h: int) -> dict[str, tuple[float, float]]:
        """{key: (revenue log-ratio, spend log-ratio)} averaged over every usable year."""
        t_ref = int((pd.Timestamp(origin) - self.dates[0]).days)   # may lie beyond the reference end
        last = len(self.dates) - 1
        cs_rev, cs_sp = _cumsum(self.revenue), _cumsum(self.spend)
        rev_vals, sp_vals = [], []
        for shift in _analog_shifts():
            r = _analog_ratio(cs_rev, t_ref, h, shift, last, None)
            s = _analog_ratio(cs_sp, t_ref, h, shift, last, None)
            if r is not None and s is not None:
                rev_vals.append(r)
                sp_vals.append(s)
        rev_mean, _ = _mean_over_years(rev_vals, len(self.keys))
        sp_mean, _ = _mean_over_years(sp_vals, len(self.keys))
        return {k: (float(rev_mean[i]), float(sp_mean[i])) for i, k in enumerate(self.keys)}


def fill_seasonal_from_reference(feats: pd.DataFrame, channels: pd.Series,
                                 reference: SeasonalReference | None) -> pd.DataFrame:
    """Fill missing portfolio/channel analog ratios from the pickled reference history.

    `channels` is the channel name of every feature row (aligned on the index).
    """
    if reference is None or feats.empty or not feats["seas_port_rev"].isna().any():
        return feats
    out = feats.copy()
    for (origin, h), idx in out.groupby(["origin", "horizon"]).groups.items():
        ratios = reference.analog(origin, int(h))
        port = ratios.get("all", (np.nan, np.nan))
        chan_rev = channels.loc[idx].map(lambda c: ratios.get(c, port)[0]).to_numpy(dtype=float)
        chan_sp = channels.loc[idx].map(lambda c: ratios.get(c, port)[1]).to_numpy(dtype=float)
        for col, val in (("seas_port_rev", port[0]), ("seas_port_sp", port[1]),
                         ("seas_chan_rev", chan_rev), ("seas_chan_sp", chan_sp)):
            cur = out.loc[idx, col].to_numpy(dtype=float)
            out.loc[idx, col] = np.where(np.isnan(cur), val, cur)
    return out


# ─── Feature builder ─────────────────────────────────────────────────────────

class FeatureBuilder:
    """Vectorised feature + target computation for every series at one origin."""

    def __init__(self, hier: Hierarchy, lag: int = MATURITY_LAG_DAYS):
        self.hier = hier
        self.lag = lag
        self.meta = hier.meta
        self.n, self.D = hier.revenue.shape

        self.cs_rev = _cumsum(hier.revenue)
        self.cs_sp = _cumsum(hier.spend)
        self.cs_zero_rev = _cumsum((hier.revenue <= 0).astype(float))
        self.cs_zero_sp = _cumsum((hier.spend <= 0).astype(float))

        idx = np.arange(self.D)[None, :]
        active = (hier.spend > 0) | (hier.revenue > 0)
        self.last_spend_idx = np.maximum.accumulate(np.where(hier.spend > 0, idx, -1), axis=1)
        self.first_active = np.where(active.any(axis=1), active.argmax(axis=1), self.D)

        future = pd.date_range(hier.dates[-1] + pd.Timedelta(days=1), periods=max(HORIZONS) + 1, freq="D")
        self.cal_dates = hier.dates.append(future)
        self.cs_peak = np.concatenate([[0.0], np.cumsum(_peak_mask(self.cal_dates).astype(float))])

        self.parent = self.meta["parent_id"].to_numpy()
        self.channel_sid = self.meta["channel_id"].to_numpy()
        self.level_code = self.meta["level_code"].to_numpy()
        self.channel_code = self.meta["channel"].map(CHANNEL_CODES).fillna(len(CHANNEL_CODES)).to_numpy(dtype=int)
        self.type_code = self.meta["campaign_type"].map(TYPE_CODES).fillna(TYPE_CODES["UNKNOWN"]).to_numpy(dtype=int)

    def _seasonal(self, cs: np.ndarray, t: int, h: int, avail_end: int,
                  c_daily: np.ndarray) -> np.ndarray:
        """Mean analog log-ratio over every usable year (NaN where no year is usable)."""
        vals = [v for shift in _analog_shifts()
                if (v := _analog_ratio(cs, t, h, shift, avail_end, c_daily)) is not None]
        return _mean_over_years(vals, cs.shape[0])[0]

    def build(self, t: int, horizons: list[int] | None = None, avail_end: int | None = None,
              with_targets: bool = True) -> pd.DataFrame:
        """Feature rows for every series at origin index `t`, one per horizon.

        Series with no activity in the last 91 days are returned with
        `alive=False`; they are forecast as zero and never used for training.

        `avail_end` is the last day index the caller is allowed to look at (used
        for targets and for leave-one-year-out analogs); it defaults to the end
        of the hierarchy.
        """
        horizons = horizons or HORIZONS
        avail_end = self.D - 1 if avail_end is None else avail_end
        if t + 1 < MIN_HISTORY_DAYS:
            return pd.DataFrame()
        e = max(t - self.lag, 0)              # last day whose revenue is treated as mature

        # Spend is final the day it is reported, so spend windows run right up to
        # the origin.  Revenue for the last `lag` days is still maturing, so those
        # days are nowcast as spend x trailing ROAS instead of trusting the export.
        if e < t:
            a_tr = max(e - BASE_WINDOW + 1, 0)
            sp_tr = _wsum(self.cs_sp, a_tr, e)
            roas_tr = np.where(sp_tr > 0, _wsum(self.cs_rev, a_tr, e) / np.maximum(sp_tr, 1e-9), 0.0)
            nowcast = _wsum(self.cs_sp, e + 1, t) * roas_tr.clip(0, NOWCAST_ROAS_CAP)
        else:
            nowcast = np.zeros(self.n)

        rev, sp = {}, {}
        for w in RUN_WINDOWS:
            a = max(t - w + 1, 0)
            n_days = t - a + 1
            rev[w] = (_wsum(self.cs_rev, a, e) + nowcast) / n_days
            sp[w] = _wsum(self.cs_sp, a, t) / n_days

        c_rev = 0.10 * rev[182] + 1.0
        c_sp = 0.10 * sp[182] + 0.25

        def lr(num, den, c):
            return np.log((num + c) / (den + c))

        f: dict[str, np.ndarray] = {}
        f["log_rev_28"] = np.log1p(rev[28] * 28)
        f["log_sp_28"] = np.log1p(sp[28] * 28)
        for w in (7, 14, 56, 91, 182):
            f[f"r{w}_28"] = lr(rev[w], rev[28], c_rev)
            f[f"s{w}_28"] = lr(sp[w], sp[28], c_sp)
        for w in (7, 28, 91):
            f[f"log_roas_{w}"] = np.log((rev[w] + c_rev) / (sp[w] + c_sp))

        par = np.where(self.parent >= 0, self.parent, 0)
        f["log_share_parent"] = np.where(
            self.parent >= 0, np.log((rev[28] + c_rev) / (rev[28][par] + c_rev)), 0.0)
        for name in ("r7_28", "r91_28", "s7_28"):
            f[f"parent_{name}"] = np.where(self.parent >= 0, f[name][par], 0.0)
            f[f"port_{name}"] = np.full(self.n, f[name][0])

        # weekly volatility over the last 8 complete, mature weeks
        if e - 55 >= 0:
            wk_rev = np.stack([_wsum(self.cs_rev, e - 7 * j - 6, e - 7 * j) for j in range(8)], axis=1)
            wk_sp = np.stack([_wsum(self.cs_sp, e - 7 * j - 6, e - 7 * j) for j in range(8)], axis=1)
            f["rev_cv_8w"] = wk_rev.std(axis=1) / (wk_rev.mean(axis=1) + c_rev * 7)
            f["sp_cv_8w"] = wk_sp.std(axis=1) / (wk_sp.mean(axis=1) + c_sp * 7)
        else:
            f["rev_cv_8w"] = np.full(self.n, np.nan)
            f["sp_cv_8w"] = np.full(self.n, np.nan)

        a_rev, a_sp = max(e - BASE_WINDOW + 1, 0), max(t - BASE_WINDOW + 1, 0)
        f["zero_rev_share_28"] = _wsum(self.cs_zero_rev, a_rev, e) / (e - a_rev + 1)
        f["zero_sp_share_28"] = _wsum(self.cs_zero_sp, a_sp, t) / (t - a_sp + 1)
        last_spend = self.last_spend_idx[:, t]
        f["log_days_since_spend"] = np.log1p(np.where(last_spend >= 0, t - last_spend, 365).clip(0, 365))
        f["log_age_days"] = np.log1p((t - self.first_active).clip(0, 1095))

        exists = self.first_active <= t
        alive = exists & ((sp[91] > 0) | (rev[91] > 0))

        frames = []
        for h in horizons:
            g = dict(f)
            seas_rev = self._seasonal(self.cs_rev, t, h, avail_end, c_rev)
            seas_sp = self._seasonal(self.cs_sp, t, h, avail_end, c_sp)
            g["seas_port_rev"] = np.full(self.n, seas_rev[0])
            g["seas_port_sp"] = np.full(self.n, seas_sp[0])
            g["seas_chan_rev"] = np.where(self.level_code >= 1, seas_rev[self.channel_sid], seas_rev[0])
            g["seas_chan_sp"] = np.where(self.level_code >= 1, seas_sp[self.channel_sid], seas_sp[0])
            g["seas_own_rev"] = seas_rev
            g["seas_own_sp"] = seas_sp
            g["seas_n_years"] = np.full(self.n, float(np.isfinite(seas_rev[0])))

            g["peak_share_target"] = np.full(self.n, (self.cs_peak[t + h + 1] - self.cs_peak[t + 1]) / h)
            g["peak_share_base"] = np.full(self.n, (self.cs_peak[t + 1] - self.cs_peak[a_sp]) / (t - a_sp + 1))
            mid = self.cal_dates[min(t + 1 + h // 2, len(self.cal_dates) - 1)].dayofyear
            g["target_doy_sin"] = np.full(self.n, np.sin(2 * np.pi * mid / 365.25))
            g["target_doy_cos"] = np.full(self.n, np.cos(2 * np.pi * mid / 365.25))

            g["horizon"] = np.full(self.n, h)
            g["level_code"] = self.level_code
            g["channel_code"] = self.channel_code
            g["type_code"] = self.type_code

            out = pd.DataFrame(g)
            out["series_id"] = self.meta["series_id"].to_numpy()
            out["origin_idx"] = t
            out["origin"] = self.hier.dates[t]
            out["alive"] = alive
            out["base_rev_daily"] = rev[28]
            out["base_sp_daily"] = sp[28]
            out["c_rev_daily"] = c_rev
            out["c_sp_daily"] = c_sp

            if with_targets and t + h <= avail_end:
                out["y_rev"] = _wsum(self.cs_rev, t + 1, t + h)
                out["y_sp"] = _wsum(self.cs_sp, t + 1, t + h)
            else:
                out["y_rev"] = np.nan
                out["y_sp"] = np.nan
            frames.append(out)

        return pd.concat(frames, ignore_index=True)


def build_samples(hier: Hierarchy, origins: list[int], horizons: list[int] | None = None) -> pd.DataFrame:
    """Feature rows with realised targets for many origins (training sample table)."""
    fb = FeatureBuilder(hier)
    frames = [fb.build(t, horizons=horizons) for t in origins]
    frames = [x for x in frames if len(x)]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[out["alive"] & out["y_rev"].notna()].reset_index(drop=True)


def latest_features(hier: Hierarchy, horizons: list[int] | None = None) -> pd.DataFrame:
    """Feature rows at the forecast origin (the last day of the hierarchy)."""
    return FeatureBuilder(hier).build(hier.n_days - 1, horizons=horizons, with_targets=False)
