"""Forecast contract: ordered quantiles, coherent hierarchy, exact output format."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from conftest import MODEL_PATH, ROOT, make_exports
from src.config import HORIZONS
from src.explain import driver_waterfall
from src.forecast import SUBMISSION_COLUMNS, forecast, to_submission
from src.model import _ridge, apply_conformal, fit_conformal
from src.pipeline import feature_table, prepare_data


def test_quantiles_are_ordered_and_non_negative(baseline):
    for kind in ("revenue", "spend", "roas"):
        p10, p50, p90 = (baseline[f"{kind}_{q}"] for q in ("p10", "p50", "p90"))
        assert (p10 >= 0).all()
        assert (p10 <= p50 + 1e-9).all() and (p50 <= p90 + 1e-9).all()
    assert baseline[["revenue_p10", "revenue_p50", "revenue_p90"]].notna().all().all()


def test_hierarchy_is_coherent_at_the_median(baseline):
    for h in HORIZONS:
        f = baseline[baseline["horizon_days"] == h]
        blended = f.loc[f["level"] == "blended", "revenue_p50"].iloc[0]
        channels = f[f["level"] == "channel"]
        types = f[f["level"] == "campaign_type"]
        np.testing.assert_allclose(channels["revenue_p50"].sum(), blended, rtol=1e-6)
        by_channel = types.groupby("channel")["revenue_p50"].sum()
        np.testing.assert_allclose(by_channel.reindex(channels["channel"]).to_numpy(),
                                   channels["revenue_p50"].to_numpy(), rtol=1e-6)
        # campaigns may sum to less than their type (future campaigns), never to more
        camp = f[f["level"] == "campaign"].groupby(["channel", "campaign_type"])["revenue_p50"].sum()
        typ = types.set_index(["channel", "campaign_type"])["revenue_p50"]
        assert (camp <= typ.reindex(camp.index) * (1 + 1e-6)).all()


def test_forecast_is_in_a_sane_range_for_a_stable_account(baseline):
    """The synthetic account is stationary, so the median should sit near the run-rate."""
    b = baseline[baseline["level"] == "blended"]
    ratio = b["revenue_p50"] / b["baseline_revenue"]
    assert ratio.between(0.6, 1.5).all()
    assert (b["roas_p50"].between(2.0, 8.0)).all()


def test_longer_horizons_forecast_more(baseline):
    b = baseline[baseline["level"] == "blended"].sort_values("horizon_days")
    assert b["revenue_p50"].is_monotonic_increasing


def test_waterfall_reproduces_the_median(baseline):
    for _, row in baseline[baseline["level"].isin(["blended", "channel"])].iterrows():
        wf = driver_waterfall(row)
        steps = wf.loc[wf["kind"] != "total", "effect"].sum()
        np.testing.assert_allclose(steps, row["revenue_p50"], rtol=1e-6)


def test_submission_format(baseline, prepared):
    sub = to_submission(baseline)
    assert list(sub.columns) == SUBMISSION_COLUMNS
    assert len(sub) == prepared.hier.n_series * len(HORIZONS)
    assert not sub.duplicated(["channel", "campaign_type", "campaign_name", "horizon_days"]).any()
    assert set(sub["horizon_days"]) == set(HORIZONS)
    assert (sub.loc[sub["channel"] == "all", "campaign_type"] == "").all()
    assert sub[[c for c in sub.columns if c.startswith("p")]].notna().all().all()


def test_forecast_is_deterministic(prepared, bundle, baseline):
    again = forecast(bundle, feature_table(prepared.hier))
    pd.testing.assert_frame_equal(again.reset_index(drop=True), baseline.reset_index(drop=True))


def test_tiny_and_single_channel_inputs(tmp_path, bundle):
    """Whatever lands in data/ must produce a complete file: 3 days of history, one channel."""
    make_exports(tmp_path / "tiny", days=3)
    fc = forecast(bundle, feature_table(prepare_data(tmp_path / "tiny").hier))
    assert to_submission(fc).notna().all().all()

    only_google = tmp_path / "google_only"
    only_google.mkdir()
    make_exports(tmp_path / "full", days=90)
    (only_google / "g.csv").write_bytes((tmp_path / "full" / "google_ads_campaign_stats.csv").read_bytes())
    fc = forecast(bundle, feature_table(prepare_data(only_google).hier))
    assert set(fc["channel"]) == {"all", "google"}


def test_run_entry_points_end_to_end(exports, tmp_path):
    """The two commands run.sh executes, on unseen data, produce the scored file."""
    feats, out = tmp_path / "features.parquet", tmp_path / "out" / "predictions.csv"
    for cmd in ([sys.executable, "src/generate_features.py", "--data-dir", str(exports), "--out", str(feats)],
                [sys.executable, "src/predict.py", "--features", str(feats), "--model", str(MODEL_PATH),
                 "--output", str(out)]):
        done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    sub = pd.read_csv(out, keep_default_na=False)
    assert list(sub.columns) == SUBMISSION_COLUMNS and len(sub) > 0


def test_ridge_without_intercept_recovers_coefficients():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3))
    y = X @ np.array([0.5, -0.2, 1.0]) + rng.normal(scale=0.01, size=500)
    coef, intercept = _ridge(X, y, np.ones(500), alpha=1e-6, intercept=False)
    np.testing.assert_allclose(coef, [0.5, -0.2, 1.0], atol=0.01)
    assert intercept == 0.0


def test_conformal_offsets_restore_coverage():
    """Quantile predictions that are far too narrow are widened to about 80% coverage."""
    rng = np.random.default_rng(2)
    n = 4000
    oos = pd.DataFrame({
        "origin": pd.Timestamp("2025-01-06") + pd.to_timedelta(rng.integers(0, 40, n) * 7, unit="D"),
        "horizon": 30, "level_group": "top", "calib_group": "blended", "alive": True,
        "z_rev": rng.normal(0, 0.5, n), "z_sp": 0.0, "z1_rev": 0.0, "z1_sp": 0.0,
    })
    for target in ("rev", "sp", "roas"):
        oos[f"q_{target}_lo"], oos[f"q_{target}_hi"] = -0.05, 0.05
    cal = apply_conformal(oos, fit_conformal(oos))
    inside = ((oos["z_rev"] >= cal["lo_rev"]) & (oos["z_rev"] <= cal["hi_rev"])).mean()
    assert 0.77 <= inside <= 0.84


def test_quantiles_never_shrink_with_horizon(baseline):
    """Cumulative totals: every quantile of a longer window is at least that of a shorter one."""
    for col in ("revenue_p10", "revenue_p90", "spend_p10", "spend_p90"):
        wide = baseline.pivot(index="series_id", columns="horizon_days", values=col)
        assert (wide[60] >= wide[30] - 1e-6).all() and (wide[90] >= wide[60] - 1e-6).all(), col
