"""AIgnition - probabilistic revenue & ROAS forecasting workbench.

    streamlit run app.py

The tabs follow an analyst's workflow: read the forecast -> drill into channels
and campaigns -> plan budgets -> read the AI interpretation -> check the data and
the model's track record.  Charts live in src/charts.py; this file is layout.
"""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit import config as st_config

from src import charts, explain, scenario, theme
from src.charts import LEVEL_LABEL, md, series_label, usd
from src.config import BUDGET_MULTIPLIER_BOUNDS, HORIZONS, INTERVAL_COVERAGE, MATURITY_LAG_DAYS
from src.forecast import public_columns, to_submission
from src.insights import available_providers, chat, generate_insights
from src.service import DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH, Workspace, load_workspace

st.set_page_config(page_title="AIgnition Forecast", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

PLOT_CONFIG = {"displayModeBar": False}
TABS = ["Forecast", "Explore", "Budget", "AI insights", "Data quality", "Model & backtest"]
SUGGESTED_QUESTIONS = [
    "Which campaign type should get the next $5,000, and why?",
    "What is the biggest risk to this forecast?",
    "Explain the downside (P10) case in plain language.",
]



# ─── Theme ───────────────────────────────────────────────────────────────────

def apply_theme() -> bool:
    """Bring Streamlit's widget theme, the stylesheet and the chart palette in line with the chosen mode.

    The mode lives in session state (seeded by ?theme=dark).  Streamlit reads its
    widget theme from config at the start of a run, so a change needs one rerun.
    """
    if "theme_dark" not in st.session_state:
        st.session_state["theme_dark"] = st.query_params.get("theme", "light") == "dark"
    dark = bool(st.session_state["theme_dark"])
    wanted = theme.STREAMLIT_THEME["dark" if dark else "light"]
    if any(st_config.get_option(f"theme.{k}") != v for k, v in wanted.items()):
        for k, v in wanted.items():
            st_config.set_option(f"theme.{k}", v)
        st.rerun()
    charts.use_theme(dark)
    st.markdown(theme.css(dark), unsafe_allow_html=True)
    return dark


def _toggle_dark() -> None:
    st.session_state["theme_dark"] = bool(st.session_state.get("dark_toggle"))


# ─── Small UI helpers ────────────────────────────────────────────────────────

def stat(col, label: str, value: str, sub: str = "", tone: str | None = None, cls: str = "",
         icon: str | None = None, viz: str = "") -> None:
    """Stat tile: label, value, one line of context, and optionally an icon and a small inline graphic.

    `tone` ('up' / 'down') colours the context line; `viz` is ready-made HTML (sparkline or range bar).
    """
    tone_cls = f" {tone}" if tone else ""
    badge = f"<span class='ico'>{theme.icon(icon)}</span>" if icon else ""
    graphic = f"<div class='viz'>{viz}</div>" if viz else ""
    size = " long" if len(value) > 11 else ""          # a range such as "$80.3k - $361k" needs a smaller face
    col.markdown(f"<div class='tile {cls}{' kpi' if viz else ''}'><div class='t-head'><span class='label'>{label}"
                 f"</span>{badge}</div><div class='value{size}'>{value}</div>"
                 f"<div class='sub{tone_cls}'>{sub}&nbsp;</div>{graphic}</div>", unsafe_allow_html=True)


def card():
    """A bordered container styled as an elevated card (see `st-key-card_` in src/theme.py)."""
    st.session_state["_card_n"] = st.session_state.get("_card_n", 0) + 1
    return st.container(border=True, key=f"card_{st.session_state['_card_n']}")


@contextmanager
def section(title: str, caption: str | None = None):
    """A card with a title and an optional one-line explanation."""
    box = card()
    with box:
        st.markdown(f"<div class='sec-title'>{title}</div>"
                    + (f"<div class='sec-cap'>{caption}</div>" if caption else ""), unsafe_allow_html=True)
        yield box


def show(fig, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOT_CONFIG, key=key)


def table_height(n_rows: int, max_height: int = 330) -> int:
    return int(min(max_height, 38 * (n_rows + 1) + 3))


def forecast_table(df: pd.DataFrame, with_horizon: bool = False):
    """Styled forecast table (the table-view twin of the charts)."""
    out = pd.DataFrame({
        "Series": df.apply(series_label, axis=1), "Horizon (days)": df["horizon_days"],
        "Revenue P10": df["revenue_p10"], "Revenue P50": df["revenue_p50"], "Revenue P90": df["revenue_p90"],
        "Spend": df["spend_p50"], "ROAS P10": df["roas_p10"], "ROAS P50": df["roas_p50"], "ROAS P90": df["roas_p90"],
        "vs run-rate": (df["revenue_p50"] / df["baseline_revenue"].replace(0, np.nan) - 1) * 100,
    })
    if not with_horizon:
        out = out.drop(columns="Horizon (days)")
    money = {c: "${:,.0f}" for c in ("Revenue P10", "Revenue P50", "Revenue P90", "Spend")}
    ratio = {c: "{:.2f}x" for c in ("ROAS P10", "ROAS P50", "ROAS P90")}
    return out.style.format({**money, **ratio, "vs run-rate": "{:+.1f}%"}, na_rep="-")


def pick(fc: pd.DataFrame, horizon: int, level: str = "blended") -> pd.Series:
    return fc[(fc["horizon_days"] == horizon) & (fc["level"] == level)].iloc[0]


# ─── Data loading ────────────────────────────────────────────────────────────

def _signature(folder: Path) -> tuple:
    files = sorted(folder.rglob("*.csv"))
    model = DEFAULT_MODEL_PATH.stat().st_mtime if DEFAULT_MODEL_PATH.exists() else 0
    return tuple((f.name, f.stat().st_size, f.stat().st_mtime) for f in files) + (model,)


@st.cache_resource(show_spinner="Reading exports, validating and forecasting…")
def get_workspace(folder: str, signature: tuple) -> Workspace:
    return load_workspace(folder, DEFAULT_MODEL_PATH)


@st.cache_data(show_spinner="Writing the interpretation…", ttl=3600)
def cached_insights(facts_json: str, use_llm: bool) -> dict:
    return generate_insights(json.loads(facts_json), use_llm=use_llm)


def uploaded_folder(files) -> Path:
    """Persist uploaded CSVs for this session and return their folder."""
    key = tuple((f.name, f.size) for f in files)
    if st.session_state.get("upload_key") != key:
        folder = Path(tempfile.mkdtemp(prefix="aignition_"))
        for f in files:
            (folder / f.name).write_bytes(f.getvalue())
        st.session_state["upload_key"] = key
        st.session_state["upload_dir"] = str(folder)
    return Path(st.session_state["upload_dir"])


def build_analysis(ws: Workspace, fc: pd.DataFrame, horizon: int, is_scenario: bool, use_llm: bool,
                   breakeven: float) -> dict:
    """Anomalies, risks, optimiser result, fact sheet and written insights for the current view."""
    anomalies = explain.detect_anomalies(ws.hier)
    risks = explain.risk_flags(fc, ws.hier, ws.report, horizon, breakeven_roas=breakeven)
    allocation = scenario.optimise_budget(ws.baseline, horizon)
    facts = explain.build_facts(ws.baseline, ws.hier, ws.report, ws.bundle, horizon,
                                scenario=fc if is_scenario else None, allocation=allocation,
                                anomalies=anomalies, risks=risks)
    insights = cached_insights(json.dumps(facts, sort_keys=True), use_llm)
    return {"anomalies": anomalies, "risks": risks, "allocation": allocation, "facts": facts, "insights": insights}


# ─── Budget plan state ───────────────────────────────────────────────────────
# A plan = one slider per channel x an optional override per campaign type, both
# expressed as multiples of the spend the model expects if nothing changes.

def _bump_editor() -> None:
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1


def _set_plan(channels: list[str], slider: float = 1.0, type_mult: dict | None = None) -> None:
    for c in channels:
        st.session_state[f"mult_{c}"] = float(slider)
    st.session_state["type_mult"] = dict(type_mult or {})
    _bump_editor()


def _allocation_multipliers(alloc: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {(r["channel"], r["campaign_type"]): float(r["spend_recommended"] / r["spend_current"])
            for _, r in alloc.iterrows() if r["spend_current"] > 0}


def current_multipliers(types: pd.DataFrame) -> dict[tuple[str, str], float]:
    overrides = st.session_state.get("type_mult", {})
    return {(c, t): float(st.session_state.get(f"mult_{c}", 1.0)) * float(overrides.get((c, t), 1.0))
            for c, t in zip(types["channel"], types["campaign_type"])}


def _save_plan(multipliers: dict) -> None:
    name = (st.session_state.get("plan_name") or "").strip() or f"Plan {len(st.session_state.get('saved_plans', [])) + 1}"
    plans = [p for p in st.session_state.get("saved_plans", []) if p["name"] != name]
    st.session_state["saved_plans"] = plans + [{"name": name, "multipliers": dict(multipliers)}]
    st.session_state["plan_name"] = ""


def _ask(question: str) -> None:
    st.session_state["pending_question"] = question


# ─── Page header ─────────────────────────────────────────────────────────────

def header(ws: Workspace, horizon: int, is_scenario: bool, insights: dict) -> None:
    origin = ws.hier.origin
    start, end = origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=horizon)
    chips = [
        f"Forecast window <b>{start.strftime('%b %d')} – {end.strftime('%b %d, %Y')}</b> ({horizon} days)",
        f"Data through <b>{origin.strftime('%b %d, %Y')}</b>",
        f"<b>{len(ws.report['channels'])}</b> channels · <b>{ws.report['n_campaigns']}</b> campaigns",
    ]
    bt = pd.DataFrame(ws.bundle.get("backtest", {}).get("by_level", []))
    if len(bt):
        b = bt[(bt["level"] == "blended") & (bt["horizon_days"] == horizon)]
        if len(b):
            chips.append(f"Track record: typical miss <b>{b.iloc[0]['wape']:.0f}%</b>, "
                         f"<b>{b.iloc[0]['wape_budget_known']:.0f}%</b> with a budget plan")
    source = insights.get("source", "rules")
    chips.append(f"Insights: <b>{insights.get('model') if source.startswith('llm') else 'rule-based'}</b>")
    html = "".join(f"<span class='chip'>{c}</span>" for c in chips)
    if is_scenario:
        html += "<span class='chip plan'><b>Budget plan active</b> · numbers reflect your plan</span>"
    st.markdown(
        f"<div class='hero'><div class='row'><div class='mark'>{theme.LOGO}</div><div>"
        "<div class='title'>Revenue and ROAS forecast</div>"
        "<div class='sub'>Probabilistic outlook for Google Ads, Microsoft Ads and Meta Ads</div></div></div>"
        f"<div class='chips'>{html}</div></div>", unsafe_allow_html=True)


def kpi_row(row: pd.Series, compare: pd.Series | None = None, history: pd.DataFrame | None = None) -> None:
    """Four headline tiles. `history` (weekly revenue / spend) adds 12-week sparklines."""
    dark = bool(st.session_state.get("theme_dark"))
    c1, c2, c3, c4 = st.columns(4)
    ref = compare["revenue_p50"] if compare is not None else row["baseline_revenue"]
    ref_name = "baseline" if compare is not None else "28-day run-rate"
    change = (row["revenue_p50"] / max(ref, 1e-9) - 1) * 100
    rev_spark = theme.sparkline(history["revenue"].tail(12), dark) if history is not None else ""
    sp_spark = theme.sparkline(history["spend"].tail(12), dark) if history is not None else ""
    stat(c1, "Revenue, most likely (P50)", usd(row["revenue_p50"]),
         f"{'▲' if change >= 0 else '▼'} {abs(change):.1f}% vs {ref_name}", "up" if change >= 0 else "down",
         icon="trend", viz=rev_spark)
    stat(c2, "Revenue range (P10 - P90)", f"{usd(row['revenue_p10'])} - {usd(row['revenue_p90'])}",
         f"{INTERVAL_COVERAGE * 100:.0f}% of outcomes expected inside", icon="range",
         viz=theme.range_bar(row["revenue_p10"], row["revenue_p50"], row["revenue_p90"]))
    if compare is not None:
        d = (row["spend_p50"] / max(compare["spend_p50"], 1e-9) - 1) * 100
        stat(c3, "Spend (planned)", usd(row["spend_p50"]), f"{d:+.1f}% vs expected", icon="spend", viz=sp_spark)
    else:
        stat(c3, "Spend (expected)", usd(row["spend_p50"]),
             f"range {usd(row['spend_p10'])} - {usd(row['spend_p90'])}", icon="spend", viz=sp_spark)
    stat(c4, "Blended ROAS (P50)", f"{row['roas_p50']:.2f}x",
         f"range {row['roas_p10']:.2f}x - {row['roas_p90']:.2f}x", icon="ratio",
         viz=theme.range_bar(row["roas_p10"], row["roas_p50"], row["roas_p90"]))
    st.write("")


# ─── Tab: Forecast ───────────────────────────────────────────────────────────

def tab_forecast(ws: Workspace, fc: pd.DataFrame, horizon: int, is_scenario: bool, breakeven: float,
                 analysis: dict) -> None:
    base = ws.baseline
    h, base_h = fc[fc["horizon_days"] == horizon], base[base["horizon_days"] == horizon]
    blended, base_blended = pick(fc, horizon), pick(base, horizon)

    st.markdown(f"<div class='keymsg'><span class='ico'>{theme.icon('bulb', 18)}</span><div>"
                f"<div class='k'>Key message</div><div class='h'>{analysis['insights']['headline']}</div>"
                "</div></div>", unsafe_allow_html=True)
    weekly = explain.weekly_history(ws.hier, 0, 14).iloc[:-1]      # the last week is still maturing
    kpi_row(blended, base_blended if is_scenario else None, history=weekly)

    cols = st.columns(len(HORIZONS))
    for col, hz in zip(cols, HORIZONS):
        r = pick(fc, hz)
        stat(col, f"Next {hz} days", usd(r["revenue_p50"]),
             f"{usd(r['revenue_p10'])} - {usd(r['revenue_p90'])} · ROAS {r['roas_p50']:.2f}x",
             cls="mini on" if hz == horizon else "mini")
    st.write("")

    left, right = st.columns(2, gap="medium")
    with left:
        per_week = 7.0 / horizon
        with section("Revenue: history and forecast",
                     f"Weekly actuals, then the forecast as a weekly average: "
                     f"{usd(base_blended['revenue_p10'] * per_week)} to {usd(base_blended['revenue_p90'] * per_week)} per week."):
            show(charts.history_forecast_chart(ws.hier, base_blended, horizon, blended if is_scenario else None),
                 "hist_blended")
    with right:
        with section("Why the forecast differs from the run-rate",
                     "Exact decomposition of the P50 into groups of model terms."):
            show(charts.waterfall_chart(blended), "wf_blended")

    left, right = st.columns(2, gap="medium")
    with left:
        with section("Cumulative outlook",
                     "Revenue accumulated by day 30, 60 and 90. Inner band P25-P75, outer band P10-P90."):
            show(charts.cone_chart(base[base["level"] == "blended"],
                                   fc[fc["level"] == "blended"] if is_scenario else None), "cone")
    with right:
        with section("Chance of hitting a revenue goal",
                     "The forecast read as a probability: the curve gives the chance of at least each amount."):
            a, b = st.columns([1, 1])
            target = a.number_input(f"Goal for the next {horizon} days ($)", min_value=0.0,
                                    value=float(round(blended["revenue_p50"], -3)), step=5000.0)
            prob = scenario.probability_at_least(blended["revenue_p10"], blended["revenue_p50"],
                                                 blended["revenue_p90"], target)
            b.markdown(f"<div class='small'>Chance of reaching the goal</div><div class='big'>{prob * 100:.0f}%</div>"
                       f"<div class='small'>{'under your budget plan' if is_scenario else 'if budgets behave as usual'}</div>",
                       unsafe_allow_html=True)
            show(charts.exceedance_chart(blended["revenue_p10"], blended["revenue_p50"], blended["revenue_p90"],
                                         target), "exceed")

    left, right = st.columns([2, 3], gap="medium")
    with left:
        with section("Revenue by channel", "Bars are P50; whiskers are the P10-P90 range."):
            show(charts.channel_bars(h, base_h if is_scenario else None), "channel_bars")
    with right:
        with section("ROAS range by campaign type",
                     "Dot is the P50, line is P10-P90. Campaign types with under 1% of spend are left out."):
            show(charts.roas_range_chart(h, breakeven), "roas_ranges")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with section("Seasonality: this year against earlier years",
                     "Weekly revenue by week of the year. The shaded weeks are the forecast window."):
            tops = ws.hier.meta[ws.hier.meta["level"].isin(["blended", "channel"])]
            labels = {int(r["series_id"]): series_label(r) for _, r in tops.iterrows()}
            sid = st.selectbox("Series", list(labels), format_func=labels.get, label_visibility="collapsed",
                               key="season_series")
            show(charts.seasonality_chart(ws.hier, sid, horizon), "season")
    with right:
        with section("Biggest moves against the run-rate",
                     "Campaign types whose forecast departs most, in dollars, from their last-28-day pace."):
            mv = explain.movers(fc, horizon)
            if mv.empty:
                st.caption("No active campaign types.")
            else:
                show(charts.movers_chart(mv), "movers")

    with st.expander("Table view: blended and channels, all horizons"):
        table = fc[fc["level"].isin(["blended", "channel"])]
        st.dataframe(forecast_table(table, with_horizon=True), hide_index=True, use_container_width=True,
                     height=table_height(len(table), 480))


# ─── Tab: Explore ────────────────────────────────────────────────────────────

def tab_explore(ws: Workspace, fc: pd.DataFrame, horizon: int) -> None:
    c1, c2 = st.columns([1, 4])
    level = c1.radio("Level", ["channel", "campaign_type", "campaign"], format_func=LEVEL_LABEL.get)
    h = fc[(fc["horizon_days"] == horizon) & (fc["level"] == level)].copy()
    if level == "campaign" and c1.checkbox("Active campaigns only", value=True):
        h = h[h["alive"] & (h["revenue_p50"] + h["spend_p50"] > 0)]
    h = h.sort_values("revenue_p50", ascending=False)
    with c2:
        st.caption(f"{len(h)} {LEVEL_LABEL[level].lower()} forecasts for the next {horizon} days. "
                   "Campaign-level numbers are directional: single campaigns start, stop and spike.")
        st.dataframe(forecast_table(h), hide_index=True, use_container_width=True, height=table_height(len(h)))
    if h.empty:
        return

    labels = {int(r["series_id"]): series_label(r) for _, r in h.iterrows()}
    sid = st.selectbox("Inspect one series", list(labels), format_func=labels.get)
    row = h[h["series_id"] == sid].iloc[0]
    name = labels[sid]

    m1, m2, m3, m4 = st.columns(4)
    change = (row["revenue_p50"] / max(row["baseline_revenue"], 1e-9) - 1) * 100
    stat(m1, "Revenue P50", usd(row["revenue_p50"]), f"{'▲' if change >= 0 else '▼'} {abs(change):.1f}% vs run-rate",
         "up" if change >= 0 else "down")
    stat(m2, "Revenue range (P10 - P90)", f"{usd(row['revenue_p10'])} - {usd(row['revenue_p90'])}")
    stat(m3, "ROAS P50", f"{row['roas_p50']:.2f}x", f"range {row['roas_p10']:.2f}x - {row['roas_p90']:.2f}x")
    stat(m4, "Spend elasticity", f"{row['elasticity']:.2f}",
         f"+10% budget gives about {((1.1 ** row['elasticity']) - 1) * 100:+.1f}% revenue")
    st.write("")

    left, right = st.columns(2, gap="medium")
    with left:
        with section(f"{name}: history and forecast"):
            show(charts.history_forecast_chart(ws.hier, row, horizon), "hist_series")
    with right:
        with section("Drivers of this forecast"):
            show(charts.waterfall_chart(row), "wf_series")

    left, right = st.columns(2, gap="medium")
    with left:
        with section(f"{name}: seasonality", "Weekly revenue by week of the year; shaded weeks are the forecast window."):
            show(charts.seasonality_chart(ws.hier, sid, horizon), "season_series_chart")
    with right:
        replay = pd.DataFrame(ws.bundle.get("backtest", {}).get("replay", []))
        if level == "channel" and len(replay) and row["channel"] in set(replay["channel"]):
            with section(f"{name}: track record",
                         f"Every past {horizon}-day forecast for this channel against what happened."):
                show(charts.replay_chart(replay, horizon, row["channel"]), "replay_series")
        else:
            with section("Cumulative outlook", "Revenue accumulated by day 30, 60 and 90."):
                show(charts.cone_chart(fc[fc["series_id"] == sid]), "cone_series")


# ─── Tab: Budget ─────────────────────────────────────────────────────────────

def plan_editor(types: pd.DataFrame, horizon: int) -> None:
    """Planned dollars per campaign type: edit in place, or upload a plan."""
    overrides = st.session_state.get("type_mult", {})
    ch_mult = np.array([float(st.session_state.get(f"mult_{c}", 1.0)) for c in types["channel"]])
    ty_mult = np.array([float(overrides.get((c, t), 1.0)) for c, t in zip(types["channel"], types["campaign_type"])])
    expected = types["spend_p50"].to_numpy()
    editor = pd.DataFrame({
        "Channel": types["channel"].str.title(), "Campaign type": types["campaign_type"],
        "Expected spend": expected, "Planned spend": np.round(expected * ch_mult * ty_mult, 2),
        "Elasticity": types["elasticity"],
    })
    edited = st.data_editor(
        editor, hide_index=True, use_container_width=True,
        disabled=["Channel", "Campaign type", "Expected spend", "Elasticity"],
        key=f"plan_editor_{st.session_state.get('editor_version', 0)}",
        column_config={"Expected spend": st.column_config.NumberColumn(format="$%.0f"),
                       "Planned spend": st.column_config.NumberColumn(format="$%.0f", min_value=0.0, step=100.0,
                                                                      help="Type the budget you intend to spend."),
                       "Elasticity": st.column_config.NumberColumn(format="%.2f")})
    planned = edited["Planned spend"].fillna(0).to_numpy(dtype=float)
    if not np.allclose(planned, editor["Planned spend"].to_numpy(), rtol=0, atol=0.5):
        base = expected * ch_mult
        new = {(c, t): float(p / b) for c, t, p, b in zip(types["channel"], types["campaign_type"], planned, base)
               if b > 0 and abs(p / b - 1.0) > 1e-6}
        st.session_state["type_mult"] = new
        _bump_editor()
        st.rerun()

    a, b = st.columns([3, 2])
    template = pd.DataFrame({"channel": types["channel"], "campaign_type": types["campaign_type"],
                             "planned_spend": np.round(expected, 2)})
    b.download_button("Download plan template (CSV)", template.to_csv(index=False), file_name="budget_plan.csv",
                      mime="text/csv", use_container_width=True)
    upload = a.file_uploader(f"Upload a budget plan for the next {horizon} days "
                             "(columns: channel, campaign_type, planned_spend)", type="csv", key="plan_upload")
    if upload is not None:
        try:
            plan = pd.read_csv(io.BytesIO(upload.getvalue()))
            plan.columns = [c.strip().lower() for c in plan.columns]
            wanted = {(str(c).strip().lower(), str(t).strip().upper()): float(s)
                      for c, t, s in zip(plan["channel"], plan["campaign_type"], plan["planned_spend"])}
            mult = scenario.multipliers_from_spend(st.session_state["_baseline"], horizon, wanted)
            st.button(f"Apply uploaded plan ({len(mult)} campaign types matched)", on_click=_set_plan,
                      args=(sorted(types["channel"].unique()), 1.0, mult), type="primary")
        except Exception as exc:
            st.error(f"Could not read the plan: {exc}")


def tab_budget(ws: Workspace, scen: pd.DataFrame, horizon: int, analysis: dict) -> None:
    base = ws.baseline
    st.session_state["_baseline"] = base
    types = scenario.type_rows(base, horizon)
    channels = sorted(types["channel"].unique())
    st.caption("Enter the media budget you plan to deploy. Budgets are multiples of the spend the model expects if "
               "nothing changes. Revenue responds through each campaign type's fitted elasticity (diminishing "
               "returns); once a budget is fixed, the range narrows to ROAS uncertainty.")

    last_year = scenario.last_year_multipliers(ws.hier, base, horizon)
    optimal = _allocation_multipliers(analysis["allocation"]) if len(analysis["allocation"]) else {}
    p = st.columns(5)
    p[0].button("Cut all 20%", on_click=_set_plan, args=(channels, 0.8), use_container_width=True)
    p[1].button("Raise all 20%", on_click=_set_plan, args=(channels, 1.2), use_container_width=True)
    p[2].button("Last year's spend, same window", on_click=_set_plan, args=(channels, 1.0, last_year),
                use_container_width=True, disabled=not last_year,
                help="Replays what each campaign type spent in this window a year ago: a starting plan for a "
                     "seasonal peak.")
    p[3].button("Optimiser's split", on_click=_set_plan, args=(channels, 1.0, optimal), use_container_width=True,
                disabled=not optimal, help="Same total budget, re-split to maximise expected revenue.")
    p[4].button("Reset to expected", on_click=_set_plan, args=(channels, 1.0), use_container_width=True)

    cols = st.columns(len(channels))
    lo, hi = BUDGET_MULTIPLIER_BOUNDS
    for col, ch in zip(cols, channels):
        expected = float(types.loc[types["channel"] == ch, "spend_p50"].sum())
        col.slider(f"{ch.title()} budget", lo, hi, step=0.05, format="%.2fx", key=f"mult_{ch}",
                   on_change=_bump_editor,
                   help=md(f"Expected spend over {horizon} days: {usd(expected, compact=False)}"))

    with st.expander("Plan in dollars by campaign type, or upload a plan"):
        plan_editor(types, horizon)

    s_h, b_h = scen[scen["horizon_days"] == horizon], base[base["horizon_days"] == horizon]
    s_bl, b_bl = pick(scen, horizon), pick(base, horizon)
    is_plan = (scen["mode"] == "scenario").any()
    kpi_row(s_bl, b_bl, history=explain.weekly_history(ws.hier, 0, 14).iloc[:-1])
    d_rev, d_sp = s_bl["revenue_p50"] - b_bl["revenue_p50"], s_bl["spend_p50"] - b_bl["spend_p50"]
    if abs(d_sp) > 1:
        st.markdown(md(f"<span class='small'>Incremental: {usd(d_rev, False)} revenue for {usd(d_sp, False)} spend "
                       f"= incremental ROAS <b>{d_rev / d_sp:.2f}x</b> (blended average {b_bl['roas_p50']:.2f}x).</span>"),
                    unsafe_allow_html=True)
    for w in (scenario.extrapolation_warnings(ws.hier, scen, horizon) if is_plan else []):
        st.warning(md(f"{w['channel'].title()} {w['campaign_type']}: planned {usd(w['planned_spend'], False)} is "
                      f"{w['ratio']:.1f}x the most it has ever spent in {horizon} days "
                      f"({usd(w['historical_max_spend'], False)}). The response curve is extrapolating."), icon="⚠️")

    left, right = st.columns([2, 3], gap="medium")
    with left:
        with section("Baseline vs plan, by channel", "Whiskers are the P10-P90 range under the plan."):
            show(charts.channel_bars(s_h, b_h), "scen_channels")
    with right:
        with section("Response curve", "How one campaign type's revenue and marginal return move with its budget."):
            labels = {int(r["series_id"]): series_label(r)
                      for _, r in types.sort_values("spend_p50", ascending=False).iterrows()}
            sid = st.selectbox("Campaign type", list(labels), format_func=labels.get, label_visibility="collapsed")
            rev_fig, mar_fig = charts.response_charts(base, scen, sid, horizon)
            a, b = st.columns(2)
            with a:
                st.markdown("<div class='small'><b>Revenue response to budget</b></div>", unsafe_allow_html=True)
                show(rev_fig, "resp_rev")
            with b:
                st.markdown("<div class='small'><b>Return on the next dollar</b></div>", unsafe_allow_html=True)
                show(mar_fig, "resp_mar")

    with section("Compare plans", "Save the current plan under a name and compare it with others and the baseline."):
        a, b, c = st.columns([2, 1, 1])
        a.text_input("Plan name", key="plan_name", placeholder="e.g. Peak push +30% Google",
                     label_visibility="collapsed")
        b.button("Save current plan", on_click=_save_plan, args=(current_multipliers(types),),
                 use_container_width=True, disabled=not is_plan)
        c.button("Clear saved plans", on_click=lambda: st.session_state.update(saved_plans=[]),
                 use_container_width=True, disabled=not st.session_state.get("saved_plans"))
        rows = [("Baseline", b_bl)]
        if is_plan:
            rows.append(("Current plan (unsaved)", s_bl))
        for plan in st.session_state.get("saved_plans", []):
            rows.append((plan["name"], pick(scenario.run_scenario(base, plan["multipliers"]), horizon)))
        saved = pd.DataFrame([{
            "Plan": name, "Spend": r["spend_p50"], "Revenue P10": r["revenue_p10"], "Revenue P50": r["revenue_p50"],
            "Revenue P90": r["revenue_p90"], "ROAS": r["roas_p50"],
            "Revenue vs baseline": (r["revenue_p50"] / max(b_bl["revenue_p50"], 1e-9) - 1) * 100,
        } for name, r in rows])
        left, right = st.columns([3, 2], gap="medium")
        with left:
            st.dataframe(saved.style.format({"Spend": "${:,.0f}", "Revenue P10": "${:,.0f}", "Revenue P50": "${:,.0f}",
                                             "Revenue P90": "${:,.0f}", "ROAS": "{:.2f}x",
                                             "Revenue vs baseline": "{:+.1f}%"}),
                         hide_index=True, use_container_width=True, height=table_height(len(saved)))
        with right:
            show(charts.scenarios_chart(saved), "saved_plans_chart")

    with section("Budget optimiser",
                 "The split of a fixed budget across campaign types that maximises expected revenue: money moves "
                 "until the return on the next dollar is equal everywhere it is allowed to move."):
        o1, o2, o3 = st.columns([1.2, 1.6, 1])
        expected_total = float(types["spend_p50"].sum())
        total = o1.number_input(f"Total budget for {horizon} days ($)", min_value=0.0,
                                value=round(expected_total, -2), step=1000.0)
        bounds = o2.slider("Each campaign type may move between", 0.25, 3.0, (0.5, 2.0), step=0.05, format="%.2fx")
        alloc = scenario.optimise_budget(base, horizon, total_budget=total, bounds=bounds)
        if alloc.empty:
            st.info("No active campaign types to optimise.")
            return
        gain = alloc["revenue_recommended"].sum() - alloc["revenue_current"].sum()
        stat(o3, "Expected revenue change", f"{'+' if gain >= 0 else '-'}{usd(abs(gain))}",
             f"{gain / max(alloc['revenue_current'].sum(), 1e-9) * 100:+.1f}% vs expected plan",
             "up" if gain >= 0 else "down")
        left, right = st.columns([3, 2], gap="medium")
        with left:
            show(charts.allocation_chart(alloc), "alloc")
        with right:
            view = pd.DataFrame({
                "Campaign type": alloc["channel"].str.title() + " · " + alloc["campaign_type"],
                "Expected": alloc["spend_current"], "Recommended": alloc["spend_recommended"],
                "Change": alloc["change_pct"], "Marginal ROAS": alloc["marginal_roas_recommended"],
                "Limit": alloc["at_bound"]})
            st.dataframe(view.style.format({"Expected": "${:,.0f}", "Recommended": "${:,.0f}",
                                            "Change": "{:+.0f}%", "Marginal ROAS": "{:.2f}x"}),
                         hide_index=True, use_container_width=True, height=table_height(len(view), 420))
            st.button("Load this allocation into the planner", on_click=_set_plan,
                      args=(channels, 1.0, _allocation_multipliers(alloc)), type="primary")


# ─── Tab: AI insights ────────────────────────────────────────────────────────

def severity_box(severity: str, text: str) -> None:
    {"high": st.error, "medium": st.warning}.get(severity, st.info)(
        md(text), icon={"high": "🔴", "medium": "🟠"}.get(severity, "🔵"))


def tab_insights(analysis: dict, use_llm: bool) -> None:
    ins, facts, risks = analysis["insights"], analysis["facts"], analysis["risks"]
    source = ins.get("source", "rules")
    grounding = ins.get("grounding", {})
    top = st.columns([5, 1])
    badges = [f"<span class='pill'>{'LLM · ' + str(ins.get('model')) if source.startswith('llm') else 'Rule-based writer'}</span>"]
    if source.startswith("llm"):
        n, bad = grounding.get("numbers_checked", 0), grounding.get("unverified", [])
        badges.append(f"<span class='pill'>{n - len(bad)}/{n} quoted numbers traced to the model's facts</span>")
    top[0].markdown("".join(badges), unsafe_allow_html=True)
    if top[1].button("Regenerate", use_container_width=True, disabled=not use_llm,
                     help="Ask the LLM again for this view."):
        cached_insights.clear()
        st.rerun()

    st.markdown(f"<div class='headline'>{ins['headline']}</div>", unsafe_allow_html=True)
    st.markdown(md(ins["executive_summary"]))
    if ins.get("note"):
        st.caption(md(ins["note"]))
    if grounding.get("unverified"):
        st.caption("Numbers the LLM derived itself (not in the fact sheet, treat as its arithmetic): "
                   + md(", ".join(grounding["unverified"])))

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("##### Why: causal drivers")
        for d in ins.get("causal_drivers", []):
            with card():
                arrow = "▲" if d.get("direction") == "up" else "▼"
                st.markdown(md(f"**{arrow} {d.get('driver', '')}** &nbsp; <span class='pill'>{d.get('basis', '')}</span>"),
                            unsafe_allow_html=True)
                st.markdown(md(f"<span class='small'>{d.get('evidence', '')}</span>"), unsafe_allow_html=True)
                st.markdown(md(d.get("likely_cause", "")))
        if ins.get("anomalies"):
            st.markdown("##### Anomalies in recent weeks")
            for a in ins["anomalies"]:
                with card():
                    st.markdown(md(f"**{a.get('what', '')}**"))
                    st.markdown(md(a.get("interpretation", "")))
                    st.markdown(md(f"<span class='small'>Next step: {a.get('action', '')}</span>"),
                                unsafe_allow_html=True)
    with right:
        st.markdown("##### Recommended actions")
        for r in ins.get("recommendations", []):
            with card():
                st.markdown(md(f"**{r.get('action', '')}**"))
                st.markdown(md(r.get("rationale", "")))
                if r.get("expected_impact"):
                    st.markdown(md(f"<span class='small'>Expected impact: {r['expected_impact']}</span>"),
                                unsafe_allow_html=True)
        st.markdown("##### Risks")
        for r in ins.get("risks", []):
            severity_box(str(r.get("severity", "low")).lower(),
                         f"**{r.get('risk', '')}**  \n{r.get('mitigation', '')}")
        st.markdown("##### How far to trust this")
        st.markdown(md(ins.get("confidence", "")))

    with st.expander("Rule-based risk flags (computed, not written by the LLM)"):
        for r in risks:
            severity_box(r["severity"], f"**{r['title']}**  \n{r['detail']}")
    with st.expander("The fact sheet the LLM was given"):
        st.json(facts, expanded=False)

    st.divider()
    st.markdown("##### Ask about this forecast")
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, q in zip(cols, SUGGESTED_QUESTIONS):
        col.button(q, on_click=_ask, args=(q,), use_container_width=True, key=f"q_{hash(q)}")
    history = st.session_state.setdefault("chat", [])
    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(md(m["content"]))
    question = st.chat_input("Ask anything about the forecast, the drivers or the budget plan")
    question = question or st.session_state.pop("pending_question", None)
    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(md(question))
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = chat(question, facts, history[:-1])
            st.markdown(md(reply["answer"]))
            if reply.get("grounding", {}).get("unverified"):
                st.caption("Derived by the LLM, not in the fact sheet: "
                           + md(", ".join(reply["grounding"]["unverified"])))
        history.append({"role": "assistant", "content": reply["answer"]})


# ─── Tab: Data quality ───────────────────────────────────────────────────────

def tab_data(ws: Workspace, anomalies: pd.DataFrame) -> None:
    rep = ws.report
    c1, c2, c3, c4 = st.columns(4)
    active = int((ws.baseline["alive"] & (ws.baseline["level"] == "campaign")
                  & (ws.baseline["horizon_days"] == HORIZONS[0])).sum())
    issues = sum(1 for c in rep["checks"] if c["status"] != "ok")
    stat(c1, "Daily rows", f"{rep['total_rows']:,}", f"{len(rep['channels'])} channels")
    stat(c2, "Campaigns", f"{rep['n_campaigns']:,}", f"{active} active in the last 91 days")
    stat(c3, "History", f"{ws.hier.n_days:,} days", f"{rep['date_min']} to {ws.hier.origin.date()}")
    stat(c4, "Checks needing attention", f"{issues} of {len(rep['checks'])}", "see the list below")
    st.write("")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with section("Validation and campaign-consistency checks"):
            icon = {"ok": "✅ OK", "warn": "⚠️ Check", "fail": "⛔ Fail"}
            checks = pd.DataFrame(rep["checks"])
            checks["Status"] = checks["status"].map(icon)
            st.dataframe(checks.rename(columns={"check": "Check", "detail": "Finding"})[["Status", "Check", "Finding"]],
                         hide_index=True, use_container_width=True, height=table_height(len(checks), 600),
                         column_config={"Status": st.column_config.TextColumn(width="small"),
                                        "Check": st.column_config.TextColumn(width="medium"),
                                        "Finding": st.column_config.TextColumn(width="large")})
    with right:
        with section("Channel coverage"):
            cov = pd.DataFrame(rep["channel_coverage"])
            cov["roas"] = cov["revenue"] / cov["spend"].replace(0, np.nan)
            cov = cov.rename(columns={"channel": "Channel", "first_date": "First day", "last_date": "Last day",
                                      "missing_days": "Missing days", "campaigns": "Campaigns", "spend": "Spend",
                                      "revenue": "Revenue", "roas": "ROAS"})
            cov = cov[["Channel", "First day", "Last day", "Missing days", "Campaigns", "Spend", "Revenue", "ROAS"]]
            st.dataframe(cov.style.format({"Spend": "${:,.0f}", "Revenue": "${:,.0f}", "ROAS": "{:.2f}x"}),
                         hide_index=True, use_container_width=True)
            st.caption("Meta's `conversion` column is read as purchase value (it is fractional and matches revenue "
                       "scale), so Meta ROAS is comparable with Google and Microsoft.")
            st.caption(f"Revenue for the final {MATURITY_LAG_DAYS} days is nowcast from spend because platforms "
                       "are still attributing conversions.")

    with section("Weekly revenue (colour) and spend (grey) by channel"):
        show(charts.small_multiples_history(ws.hier), "sm_history")

    with section("Weeks that broke pattern (last 26 weeks)",
                 "A week is flagged when revenue, spend or ROAS sits more than 3 robust standard deviations from "
                 "the median of its previous 8 weeks. Only series with at least 3% of portfolio spend are screened."):
        if anomalies.empty:
            st.success("No material anomalies in the last 26 weeks.", icon="✅")
        else:
            view = pd.DataFrame({
                "Week ending": anomalies["week_ending"],
                "Where": (anomalies["channel"].str.title() + " "
                          + anomalies["campaign_type"].str.replace("_", " ").str.title()).str.strip(),
                "Metric": anomalies["metric"].str.title(), "Direction": anomalies["direction"],
                "Value": anomalies["value"], "Typical": anomalies["typical"],
                "Revenue impact": anomalies["impact_usd"], "Robust z": anomalies["robust_z"]})
            st.dataframe(view.head(25).style.format({
                "Value": "{:,.2f}", "Typical": "{:,.2f}", "Robust z": "{:+.1f}",
                "Revenue impact": lambda v: f"{'-' if v < 0 else '+'}${abs(v):,.0f}"}),
                hide_index=True, use_container_width=True, height=table_height(min(len(view), 25), 420))


# ─── Tab: Model & backtest ───────────────────────────────────────────────────

def tab_model(ws: Workspace, horizon: int) -> None:
    bt = ws.bundle.get("backtest", {})
    if not bt:
        st.info("This model bundle has no stored backtest. Run `python -m src.train`.")
        return
    by_level = pd.DataFrame(bt["by_level"])
    st.caption(f"Rolling-origin backtest: at each of {bt['n_origins']} weekly origins ({bt['first_origin']} to "
               f"{bt['last_origin']}) the data was cut at that date, the model refitted on what was visible, and the "
               f"following 30/60/90 days forecast and scored. WAPE = total absolute error / total actual revenue.")
    b = by_level[(by_level["level"] == "blended") & (by_level["horizon_days"] == horizon)].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    stat(c1, f"Blended revenue error (WAPE), {horizon}d", f"{b['wape']:.1f}%",
         f"run-rate baseline: {b['wape_run_rate']:.1f}%")
    stat(c2, "Error when the budget is known", f"{b['wape_budget_known']:.1f}%",
         "same forecasts, re-run with actual spend")
    stat(c3, "Revenue P10-P90 coverage", f"{b['coverage']:.0f}%", f"target {INTERVAL_COVERAGE * 100:.0f}%")
    stat(c4, "ROAS P10-P90 coverage", f"{b['roas_coverage']:.0f}%", f"target {INTERVAL_COVERAGE * 100:.0f}%")
    st.write("")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with section(f"Forecast error by level, {horizon}-day horizon", "Lower is better."):
            show(charts.accuracy_chart(by_level, horizon), "acc")
    with right:
        if bt.get("vs_v31"):
            with section("Against the previous pipeline", "Same origins, same series, both refitted at every origin."):
                show(charts.legacy_chart(pd.DataFrame(bt["vs_v31"]), horizon), "legacy")

    replay = pd.DataFrame(bt["replay"])
    with section(f"Backtest replay: every past {horizon}-day forecast against what happened",
                 "The Nov-Dec peak is a budget decision: with the budget known (orange) that miss largely "
                 "disappears. The drop after the May 2025 promotion was a collapse in revenue per dollar, which a "
                 "budget plan does not fix."):
        names = {"all": "All channels", **{c: c.title() for c in sorted(set(replay["channel"]) - {"all"})}}
        channel = st.radio("Series", list(names), format_func=names.get, horizontal=True,
                           label_visibility="collapsed", key="replay_channel")
        show(charts.replay_chart(replay, horizon, channel), "replay")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with section("Are the ranges honest?", "Share of outcomes that fell inside the P10-P90 range."):
            show(charts.coverage_chart(by_level), "coverage")
    with right:
        with section("How it works"):
            st.markdown(f"""
- **Target**: the log-ratio of the next 30/60/90-day total to the last-28-day run-rate, for every node
  (blended, channel, campaign type, campaign).
- **Stage 1, median**: a ridge model with no drift term, fitted per horizon and hierarchy level, on momentum
  beyond a noise dead-zone, longer-run level, ROAS drift and prior-year analog ratios.
- **Stage 2, range**: LightGBM quantile models on stage-1 residuals.
- **Stage 3, calibration**: conformal offsets from out-of-sample residuals, per level.
- **Budget response**: revenue elasticity to spend estimated from window-over-window changes.
- **Conversion lag**: revenue for the final {MATURITY_LAG_DAYS} days is nowcast from spend x trailing ROAS.
""")
    with st.expander("Table view: full backtest"):
        st.dataframe(by_level.round(1), hide_index=True, use_container_width=True)
    with st.expander("Assumptions and limits"):
        st.markdown("""
- Platform-attributed revenue is taken as the source of truth; no attribution modelling or MMM.
- Meta `conversion` is treated as purchase value. If it is a count, set `META_CONVERSION_AS_REVENUE = False`.
- Without a budget plan, the model has to guess future spend. That guess is the largest avoidable source of error.
- The elasticity describes how revenue moved with spend in this account's history. It is not a controlled
  experiment; plans far outside historical spend are flagged as extrapolation.
- Campaign-level forecasts are directional. Budget at channel or campaign-type level.
- Two peak seasons of history: peak-season forecasts lean on prior-year analogs and have wide ranges.
""")


# ─── Exports ─────────────────────────────────────────────────────────────────

def client_brief(ws: Workspace, fc: pd.DataFrame, horizon: int, analysis: dict) -> str:
    ins, h = analysis["insights"], fc[fc["horizon_days"] == horizon]
    b = pick(fc, horizon)
    lines = [f"# Revenue forecast: next {horizon} days",
             f"_Forecast issued after {ws.hier.origin.date()}; window {analysis['facts']['window']}._", "",
             f"**{ins['headline']}**", "", ins["executive_summary"], "",
             "| | P10 | P50 | P90 |", "|---|---|---|---|",
             f"| Revenue | {usd(b['revenue_p10'], False)} | {usd(b['revenue_p50'], False)} | {usd(b['revenue_p90'], False)} |",
             f"| ROAS | {b['roas_p10']:.2f} | {b['roas_p50']:.2f} | {b['roas_p90']:.2f} |",
             f"| Spend | | {usd(b['spend_p50'], False)} | |", "", "## By channel", "",
             "| Channel | Revenue P10 | Revenue P50 | Revenue P90 | ROAS P50 |", "|---|---|---|---|---|"]
    for _, r in h[h["level"] == "channel"].iterrows():
        lines.append(f"| {r['channel'].title()} | {usd(r['revenue_p10'], False)} | {usd(r['revenue_p50'], False)} | "
                     f"{usd(r['revenue_p90'], False)} | {r['roas_p50']:.2f} |")
    lines += ["", "## Recommended actions", ""]
    lines += [f"- **{r.get('action', '')}** {r.get('rationale', '')}" for r in ins.get("recommendations", [])]
    lines += ["", "## Risks", ""]
    lines += [f"- ({r.get('severity', '')}) {r.get('risk', '')}" for r in ins.get("risks", [])]
    lines += ["", "## Reliability", "", ins.get("confidence", "")]
    return "\n".join(lines)


GLOSSARY = """
- **P50**: the most likely outcome; half of outcomes land above, half below.
- **P10 – P90**: the range expected to contain 80% of outcomes. Plan against it, not the midpoint.
- **ROAS**: revenue divided by ad spend over the window.
- **Run-rate**: the last 28 days' pace, extended over the window.
- **Elasticity**: how much of a budget change passes through to revenue (0.85 = +10% budget, about +8.4% revenue).
- **Marginal ROAS**: what the *next* dollar is expected to return, as opposed to the average.
- **WAPE**: total absolute forecast error divided by total actual revenue.
"""


# ─── Page ────────────────────────────────────────────────────────────────────

def main() -> None:
    dark = apply_theme()
    st.session_state["_card_n"] = 0
    side = st.sidebar
    side.markdown(f"<div class='brand'><div class='mark'>{theme.LOGO}</div><div><div class='name'>AIgnition</div>"
                  "<div class='tag'>Forecast, plan and explain paid media</div></div></div>", unsafe_allow_html=True)
    side.toggle("Dark mode", value=dark, key="dark_toggle", on_change=_toggle_dark)

    if not DEFAULT_MODEL_PATH.exists():
        st.error("No trained model found at pickle/model.pkl. Run `python -m src.train` first.")
        st.stop()

    side.markdown("<div class='overline'>1 · Data</div>", unsafe_allow_html=True)
    source = side.radio("Data", ["Project data folder", "Upload exports"], label_visibility="collapsed")
    folder = DEFAULT_DATA_DIR
    if source == "Upload exports":
        files = side.file_uploader("Google Ads, Microsoft Ads and Meta Ads CSV exports", type="csv",
                                   accept_multiple_files=True)
        if not files:
            st.markdown(f"<div class='hero'><div class='row'><div class='mark'>{theme.LOGO}</div><div>"
                        "<div class='title'>Upload your exports</div>"
                        "<div class='sub'>Add one or more platform CSVs in the sidebar</div></div></div></div>",
                        unsafe_allow_html=True)
            st.info("Files are recognised by their columns, so names do not matter: `segments_date` (Google Ads), "
                    "`TimePeriod` (Microsoft Ads), `date_start` (Meta Ads). Any number of rows, any date range.")
            st.stop()
        folder = uploaded_folder(files)

    try:
        ws = get_workspace(str(folder), _signature(Path(folder)))
    except Exception as exc:      # surface ingestion problems to the analyst instead of a stack trace
        st.error(f"Could not read the data: {exc}")
        st.stop()

    side.markdown("<div class='overline'>2 · Planning</div>", unsafe_allow_html=True)
    horizon = side.radio("Planning horizon", HORIZONS, format_func=lambda h: f"{h} days", horizontal=True)
    breakeven = side.number_input("Break-even ROAS", min_value=0.0, max_value=20.0, value=1.0, step=0.25,
                                  help="Campaign types expected below this ROAS are flagged as risks.")
    side.markdown("<div class='overline'>3 · Interpretation</div>", unsafe_allow_html=True)
    providers = available_providers()
    use_llm = side.toggle("AI narrative (LLM)", value=bool(providers), disabled=not providers,
                          help="Off: the same sections are written by rules from the same facts.")
    side.caption(f"LLM provider: {', '.join(providers)}" if providers
                 else "No LLM key in .env; insights are rule-based.")

    types = scenario.type_rows(ws.baseline, horizon)
    for ch in types["channel"].unique():
        st.session_state.setdefault(f"mult_{ch}", 1.0)
    multipliers = current_multipliers(types)
    is_scenario = any(abs(m - 1.0) > 1e-9 for m in multipliers.values())
    fc = scenario.run_scenario(ws.baseline, multipliers) if is_scenario else ws.baseline
    analysis = build_analysis(ws, fc, horizon, is_scenario, use_llm, breakeven)

    header(ws, horizon, is_scenario, analysis["insights"])
    tabs = st.tabs(TABS)
    with tabs[0]:
        tab_forecast(ws, fc, horizon, is_scenario, breakeven, analysis)
    with tabs[1]:
        tab_explore(ws, fc, horizon)
    with tabs[2]:
        tab_budget(ws, fc, horizon, analysis)
    with tabs[3]:
        tab_insights(analysis, use_llm)
    with tabs[4]:
        tab_data(ws, analysis["anomalies"])
    with tabs[5]:
        tab_model(ws, horizon)

    side.markdown("<div class='overline'>Export</div>", unsafe_allow_html=True)
    side.download_button("Forecast file (submission format)", to_submission(fc).to_csv(index=False),
                         file_name="predictions.csv", mime="text/csv", use_container_width=True)
    side.download_button("Detailed forecast (all levels)", public_columns(fc).to_csv(index=False),
                         file_name="forecast_detail.csv", mime="text/csv", use_container_width=True)
    side.download_button("Client brief (Markdown)", client_brief(ws, fc, horizon, analysis),
                         file_name=f"forecast_brief_{horizon}d.md", mime="text/markdown", use_container_width=True)
    with side.expander("Glossary"):
        st.markdown(GLOSSARY)
    trained = ws.bundle.get("trained_on", {})
    side.caption(f"Model v{ws.bundle.get('version')} · trained on data to {trained.get('last_date', '?')}")
    st.markdown("<div class='foot'>AIgnition · ranges are P10-P90 · forecasts are not guarantees</div>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
