"""Budget scenarios, optimiser, explanations and the insight layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import explain, scenario
from src.insights import check_grounding, generate_insights, rule_based_insights
from src.service import parse_multipliers


def _blended(fc: pd.DataFrame, h: int = 30) -> pd.Series:
    return fc[(fc["level"] == "blended") & (fc["horizon_days"] == h)].iloc[0]


def test_plan_equal_to_expected_spend_changes_nothing(baseline):
    same = scenario.run_scenario(baseline, {})
    np.testing.assert_allclose(same["revenue_p50"], baseline["revenue_p50"])
    np.testing.assert_allclose(same["spend_p50"], baseline["spend_p50"])


def test_more_budget_means_more_revenue_at_lower_roas(baseline):
    up = scenario.run_scenario(baseline, {}, default=1.5)
    down = scenario.run_scenario(baseline, {}, default=0.5)
    base, hi, lo = _blended(baseline), _blended(up), _blended(down)
    assert lo["revenue_p50"] < base["revenue_p50"] < hi["revenue_p50"]
    np.testing.assert_allclose(hi["spend_p50"], 1.5 * base["spend_p50"], rtol=1e-6)
    # diminishing returns: +50% spend buys less than +50% revenue
    assert hi["revenue_p50"] / base["revenue_p50"] < 1.5
    assert hi["roas_p50"] <= base["roas_p50"] + 1e-9


def test_budget_locked_range_is_not_wider_on_the_downside(baseline):
    """Fixing the budget removes spend uncertainty, so the low end cannot get worse."""
    locked = scenario.run_scenario(baseline, {("google", "*"): 1.0001})
    assert _blended(locked)["revenue_p10"] >= _blended(baseline)["revenue_p10"] * 0.999


def test_scenario_hierarchy_adds_up(baseline):
    scen = scenario.run_scenario(baseline, {("google", "SEARCH"): 2.0, ("meta", "*"): 0.5})
    h = scen[scen["horizon_days"] == 30]
    types = h[h["level"] == "campaign_type"]
    np.testing.assert_allclose(types["revenue_p50"].sum(), _blended(scen)["revenue_p50"], rtol=1e-6)
    np.testing.assert_allclose(types["spend_p50"].sum(), _blended(scen)["spend_p50"], rtol=1e-6)


def test_response_curve_is_increasing_and_concave(baseline):
    sid = int(scenario.type_rows(baseline, 30).sort_values("spend_p50").iloc[-1]["series_id"])
    curve = scenario.response_curve(baseline, sid, 30)
    assert curve["revenue_p50"].is_monotonic_increasing
    assert curve["marginal_roas"].is_monotonic_decreasing


def test_optimiser_respects_budget_and_bounds(baseline):
    alloc = scenario.optimise_budget(baseline, 30, bounds=(0.5, 2.0))
    np.testing.assert_allclose(alloc["spend_recommended"].sum(), alloc["spend_current"].sum(), rtol=1e-4)
    ratio = alloc["spend_recommended"] / alloc["spend_current"]
    assert ratio.between(0.5 - 1e-6, 2.0 + 1e-6).all()
    assert alloc["revenue_recommended"].sum() >= alloc["revenue_current"].sum() - 1e-6


def test_optimiser_scales_to_a_given_budget(baseline):
    current = scenario.type_rows(baseline, 30)["spend_p50"].sum()
    alloc = scenario.optimise_budget(baseline, 30, total_budget=1.3 * current)
    np.testing.assert_allclose(alloc["spend_recommended"].sum(), 1.3 * current, rtol=1e-4)


def test_goal_probability_is_a_probability():
    p = [scenario.probability_at_least(80, 100, 130, t) for t in (50, 80, 100, 130, 200)]
    assert p == sorted(p, reverse=True)
    np.testing.assert_allclose(p[1:4], [0.9, 0.5, 0.1], atol=1e-6)
    assert scenario.probability_at_least(0, 0, 0, 10) == 0.0


def test_multiplier_parsing():
    assert parse_multipliers({"google|search": 1.2, "Meta": 0.8}) == {("google", "SEARCH"): 1.2, ("meta", "*"): 0.8}


def test_anomaly_detection_finds_a_planted_collapse(prepared):
    hier = prepared.hier.truncate(prepared.hier.origin)
    hier.revenue = hier.revenue.copy()
    google = hier.meta.index[hier.meta["channel"] == "google"]
    hier.revenue[google, -21:-14] *= 0.05                 # one week of near-zero Google revenue
    found = explain.detect_anomalies(hier)
    hit = found[(found["channel"] == "google") & (found["direction"] == "below")]
    assert len(hit) > 0


def test_fact_sheet_and_rule_based_insights(baseline, prepared, bundle):
    report = prepared.report.to_dict()
    risks = explain.risk_flags(baseline, prepared.hier, report, 30)
    alloc = scenario.optimise_budget(baseline, 30)
    facts = explain.build_facts(baseline, prepared.hier, report, bundle, 30, allocation=alloc,
                                anomalies=explain.detect_anomalies(prepared.hier), risks=risks)
    assert facts["blended"]["revenue_p50"] == round(_blended(baseline)["revenue_p50"])
    assert any(r["title"] == "Short history" for r in risks)        # 150 days of synthetic data

    for writer in (rule_based_insights(facts), generate_insights(facts)):   # LLM disabled in tests
        assert writer["source"] == "rules"
        for key in ("headline", "executive_summary", "causal_drivers", "risks", "recommendations", "confidence"):
            assert writer[key]


def test_grounding_check_separates_real_from_invented_numbers():
    facts = {"blended": {"revenue_p50": 251496.0, "roas_p50": 4.99, "vs_run_rate_pct": -30.6}}
    ok = check_grounding(["Revenue of $251k at a ROAS of 4.99, 30.6% below run-rate over 30 days."], facts)
    assert ok["numbers_checked"] == 3 and ok["unverified"] == []
    bad = check_grounding(["Revenue will reach $412k with ROAS 7.3."], facts)
    assert set(bad["unverified"]) == {"$412k", "7.3"}


def test_theme_roles_and_inline_graphics():
    from src import charts, theme

    assert set(theme.ROLES["light"]) == set(theme.ROLES["dark"])
    assert set(charts._THEMES["light"]) == set(charts._THEMES["dark"])
    for dark in (False, True):
        sheet = theme.css(dark)
        assert "__VARS__" not in sheet and theme.ROLES["dark" if dark else "light"]["card"] in sheet
        assert theme.sparkline([1, 3, 2, 5], dark).startswith("<svg")
    assert theme.sparkline([1.0], False) == ""
    assert "left:" in theme.range_bar(10, 50, 100)
    charts.use_theme(False)
