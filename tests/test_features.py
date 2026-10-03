"""Features must be causal: nothing after the forecast origin may influence them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, FeatureBuilder, SeasonalReference, build_samples, latest_features
from src.model import _dead_zone, design


def test_features_ignore_the_future(prepared):
    """Features at origin t are identical whether or not later data exists."""
    hier = prepared.hier
    t = hier.n_days - 40
    truncated = hier.truncate(hier.dates[t])
    from_full = FeatureBuilder(hier).build(t, avail_end=t, with_targets=False)
    from_past = FeatureBuilder(truncated).build(t, with_targets=False)
    pd.testing.assert_frame_equal(from_full[FEATURE_COLS], from_past[FEATURE_COLS])


def test_targets_are_the_following_days(prepared):
    hier = prepared.hier
    t, h = hier.n_days - 61, 30
    rows = FeatureBuilder(hier).build(t, horizons=[h])
    expected = hier.revenue[:, t + 1:t + 1 + h].sum(axis=1)
    np.testing.assert_allclose(rows.sort_values("series_id")["y_rev"].to_numpy(), expected)


def test_training_samples_only_use_complete_windows(prepared):
    hier = prepared.hier
    samples = build_samples(hier, list(range(hier.n_days - 8, 40, -7)))
    assert samples["y_rev"].notna().all()
    assert (samples["origin_idx"] + samples["horizon"] <= hier.n_days - 1).all()
    assert samples["alive"].all()


def test_recent_revenue_is_nowcast_not_trusted(prepared):
    """Zeroing revenue on the last two days (immature conversions) must not move the run-rate."""
    hier = prepared.hier
    lagged = hier.truncate(hier.origin)
    lagged.revenue = lagged.revenue.copy()
    lagged.revenue[:, -2:] = 0.0
    a = latest_features(hier, [30]).set_index("series_id")["base_rev_daily"]
    b = latest_features(lagged, [30]).set_index("series_id")["base_rev_daily"]
    np.testing.assert_allclose(a, b)


def test_dead_zone_removes_small_moves():
    x = pd.Series([-1.0, -0.2, 0.0, 0.3, 0.9])
    np.testing.assert_allclose(_dead_zone(x, 0.35), [-0.65, 0.0, 0.0, 0.0, 0.55])


def test_short_history_uses_seasonal_reference(prepared, bundle):
    """With under a year of data the analog ratios come from the reference stored in the bundle."""
    from src.features import fill_seasonal_from_reference
    from src.pipeline import feature_table

    feats = feature_table(prepared.hier)
    assert feats["seas_port_rev"].isna().all()          # 150 days of history: no own analog year
    reference = SeasonalReference.from_dict(bundle["seasonal_reference"])
    filled = fill_seasonal_from_reference(feats, feats["channel"], reference)
    assert filled["seas_port_rev"].notna().all()


def test_design_adds_targets_only_when_known(prepared):
    feats = latest_features(prepared.hier)
    d = design(feats)
    assert {"B_rev", "C_rev", "mom_r7", "seas_peak", "level_group"} <= set(d.columns)
    assert d["z_rev"].isna().all()                      # the future is unknown at the forecast origin
