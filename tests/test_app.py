"""Headless smoke test of the dashboard, including budget planning."""

from __future__ import annotations

import pytest

from conftest import MODEL_PATH, ROOT

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture(scope="module")
def app():
    if not MODEL_PATH.exists() or not any((ROOT / "data").glob("*.csv")):
        pytest.skip("needs pickle/model.pkl and sample data")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()
    return at


def _plan_active(app) -> bool:
    return any("Budget plan active" in m.value for m in app.markdown)


def _button(app, label: str):
    return [b for b in app.button if label in b.label][0]


def test_dashboard_renders_every_tab(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Forecast", "Explore", "Budget", "AI insights", "Data quality",
                                           "Model & backtest"]
    assert not _plan_active(app)


def test_budget_slider_creates_a_plan_and_reset_clears_it(app):
    app.slider(key="mult_google").set_value(1.5).run()
    assert not app.exception and _plan_active(app)
    _button(app, "Reset to expected").click().run()
    assert not app.exception and not _plan_active(app)


def test_presets_and_saved_plans(app):
    _button(app, "Raise all 20%").click().run()
    assert not app.exception and _plan_active(app)
    _button(app, "Save current plan").click().run()
    assert not app.exception
    assert len(app.session_state["saved_plans"]) == 1
    _button(app, "Reset to expected").click().run()
    assert not _plan_active(app)


def test_optimiser_allocation_can_be_loaded(app):
    _button(app, "Load this allocation").click().run()
    assert not app.exception and _plan_active(app)
    _button(app, "Reset to expected").click().run()


def test_dark_mode_switch(app):
    from src import charts, theme

    app.toggle(key="dark_toggle").set_value(True).run()
    assert not app.exception
    assert app.session_state["theme_dark"] is True
    assert charts.ACCENT == charts._THEMES["dark"]["ACCENT"]
    assert any(theme.ROLES["dark"]["page"] in m.value for m in app.markdown)      # dark stylesheet injected
    app.toggle(key="dark_toggle").set_value(False).run()
    assert not app.exception
    assert app.session_state["theme_dark"] is False
    assert charts.ACCENT == charts._THEMES["light"]["ACCENT"]
